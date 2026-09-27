from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import time
import threading
from datetime import datetime
import numpy as np
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ------------------------------------------------------------
# CAMERA FILES
# ------------------------------------------------------------
CAMERA_SOURCE_CANDIDATES = {
    "1": ["cam1.mp4", "camera1.mp4", "cam_1.mp4", "test.mp4"],
    "2": ["cam2.mp4", "camera2.mp4", "cam_2.mp4", "test2.mp4", "test_2.mp4"]
}


def find_camera_source(cam_id):
    for filename in CAMERA_SOURCE_CANDIDATES[cam_id]:
        path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(path):
            return path

    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in {"venv", "__pycache__", ".git"}]
        for filename in files:
            if not filename.lower().endswith(".mp4"):
                continue
            low = filename.lower()
            if cam_id == "1" and "cam2" in low:
                continue
            if cam_id == "2" and "cam1" in low:
                continue
            return os.path.join(root, filename)
    return None


CAMERA_SOURCES = {
    "1": find_camera_source("1"),
    "2": find_camera_source("2")
}

CAMERA_NAMES = {
    "1": "Main Sector",
    "2": "Night Vision Asset"
}

# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
model = YOLO(MODEL_PATH)
model_lock = threading.Lock()

TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
CONFIDENCE_THRESHOLD = 0.15
INFERENCE_SIZE = 640
DETECTION_INTERVAL = 0.25
ALERT_COOLDOWN = 3.0

# ------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------
DETECTION_ENABLED = True
ALARM_ENABLED = False
NIGHT_MODE = False

# ------------------------------------------------------------
# ZONES / ALERTS
# ------------------------------------------------------------
ZONE_POINTS = {"1": [], "2": []}
zone_lock = threading.Lock()

recent_alerts = []
alert_lock = threading.Lock()
last_alert_time = {}


def zone_ready(cam_id):
    with zone_lock:
        return len(ZONE_POINTS.get(cam_id, [])) >= 3


def get_zone_points(cam_id):
    with zone_lock:
        return list(ZONE_POINTS.get(cam_id, []))


def clear_camera_alerts(cam_id):
    with alert_lock:
        camera_name = f"CAM-{int(cam_id):02d}"
        recent_alerts[:] = [a for a in recent_alerts if a.get("camera") != camera_name]
        for key in list(last_alert_time):
            if key[0] == cam_id:
                del last_alert_time[key]


def trigger_alarm():
    if not ALARM_ENABLED or winsound is None:
        return
    try:
        winsound.Beep(2500, 700)
    except Exception:
        pass


def save_evidence(frame, cam_id, class_name):
    try:
        now = datetime.now()
        filename = f"CAM-{int(cam_id):02d}_{class_name}_{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        path = os.path.join(EVIDENCE_DIR, filename)
        cv2.imwrite(path, frame)
        return filename
    except Exception as error:
        print("Evidence save error:", error)
        return None


def create_alert(cam_id, class_name, confidence, frame):
    if not zone_ready(cam_id) or not DETECTION_ENABLED:
        return

    now = time.monotonic()
    key = (cam_id, class_name)

    with alert_lock:
        if now - last_alert_time.get(key, 0) < ALERT_COOLDOWN:
            return

        if not zone_ready(cam_id):
            return

        last_alert_time[key] = now
        evidence = save_evidence(frame, cam_id, class_name)

        recent_alerts.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "camera": f"CAM-{int(cam_id):02d}",
            "event": f"{class_name.upper()} entered restricted zone",
            "severity": "CRITICAL",
            "confidence": round(confidence * 100, 1),
            "evidence": evidence
        })

        if len(recent_alerts) > 50:
            recent_alerts.pop()

    threading.Thread(target=trigger_alarm, daemon=True).start()


# ------------------------------------------------------------
# ZONE GEOMETRY
# ------------------------------------------------------------
def zone_polygon_for_frame(frame, cam_id):
    points = get_zone_points(cam_id)
    if len(points) < 3:
        return None

    height, width = frame.shape[:2]
    pixel_points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in points]
    return np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))


def draw_zone(frame, cam_id):
    points = get_zone_points(cam_id)
    if not points:
        return

    height, width = frame.shape[:2]
    pixel_points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in points]
    pts = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))

    if len(pixel_points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
    else:
        cv2.polylines(frame, [pts], False, (0, 0, 255), 3)
        for point in pixel_points:
            cv2.circle(frame, point, 6, (0, 0, 255), -1)


def bbox_touches_zone(polygon, x1, y1, x2, y2):
    """Robust zone test for people and vehicles.

    A detection is considered inside if:
    - its center is inside, OR
    - its bottom-center is inside, OR
    - any bbox corner is inside, OR
    - enough of the bbox overlaps the polygon.
    """
    if polygon is None:
        return False

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    bottom = ((x1 + x2) // 2, y2)
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    for point in [center, bottom] + corners:
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return True

    # Calculate actual polygon/bounding-box overlap on a small local mask.
    px, py, pw, ph = cv2.boundingRect(polygon)
    ix1, iy1 = max(x1, px), max(y1, py)
    ix2, iy2 = min(x2, px + pw), min(y2, py + ph)

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    roi_w = ix2 - ix1
    roi_h = iy2 - iy1
    local_polygon = polygon.copy()
    local_polygon[:, 0, 0] -= ix1
    local_polygon[:, 0, 1] -= iy1

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)

    overlap_pixels = cv2.countNonZero(mask)
    bbox_pixels = max(1, (x2 - x1) * (y2 - y1))
    overlap_ratio = overlap_pixels / bbox_pixels

    return overlap_ratio >= 0.02


# ------------------------------------------------------------
# YOLO DETECTION
# ------------------------------------------------------------
def model_class_name(class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id)).lower()
    if 0 <= class_id < len(names):
        return str(names[class_id]).lower()
    return str(class_id).lower()


def detect_in_restricted_zone(frame, cam_id):
    if not DETECTION_ENABLED or not zone_ready(cam_id):
        return [], 0

    polygon = zone_polygon_for_frame(frame, cam_id)
    if polygon is None:
        return [], 0

    detections = []
    raw_count = 0

    try:
        # IMPORTANT: Do not pass a classes list here.
        # We filter classes after YOLO returns its results. This avoids
        # model-version/name-map issues that can otherwise result in zero detections.
        with model_lock:
            results = model.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                device="cpu"
            )

        for result in results:
            if result.boxes is None:
                continue

            raw_count += len(result.boxes)

            for box in result.boxes:
                try:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model_class_name(class_id)
                    if class_name not in TARGET_CLASSES:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1 = max(0, min(frame.shape[1] - 1, x1))
                    y1 = max(0, min(frame.shape[0] - 1, y1))
                    x2 = max(0, min(frame.shape[1] - 1, x2))
                    y2 = max(0, min(frame.shape[0] - 1, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    if not bbox_touches_zone(polygon, x1, y1, x2, y2):
                        continue

                    detections.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "class_name": class_name,
                        "confidence": confidence
                    })
                except Exception as box_error:
                    print(f"CAM-{int(cam_id):02d} box error:", box_error)

    except Exception as error:
        print(f"CAM-{int(cam_id):02d} YOLO error: {type(error).__name__}: {error}")

    return detections, raw_count


# ------------------------------------------------------------
# CAMERA WORKER
# ------------------------------------------------------------
class CameraWorker:
    def __init__(self, cam_id, source_path):
        self.cam_id = cam_id
        self.source_path = source_path
        self.running = True

        self.raw_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.detection_lock = threading.Lock()

        self.latest_raw = None
        self.latest_jpeg = None
        self.latest_detections = []
        self.last_raw_count = 0
        self.last_in_zone_count = 0
        self.last_inference_ok = False
        self.last_error = ""
        self.last_inference_time = 0.0
        self.sequence = 0
        self.fps = 25.0
        self.condition = threading.Condition(self.output_lock)

        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def create_status_frame(self, message):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"CAM-{int(self.cam_id):02d}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (79, 195, 247), 3)
        cv2.putText(frame, message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 190, 190), 2)
        return frame

    def open_video(self):
        if not self.source_path:
            return None
        print(f"CAM-{self.cam_id} source: {self.source_path}")
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 25.0
        self.fps = fps
        return cap

    def publish_frame(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.condition:
            self.latest_jpeg = buffer.tobytes()
            self.sequence += 1
            self.condition.notify_all()

    def capture_loop(self):
        cap = None
        next_frame_time = time.monotonic()

        while self.running:
            if cap is None:
                cap = self.open_video()
                if cap is None:
                    self.publish_frame(self.create_status_frame("Camera video file not found"))
                    time.sleep(2)
                    continue
                next_frame_time = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_frame_time = time.monotonic()
                continue

            with self.raw_lock:
                self.latest_raw = frame.copy()

            with self.detection_lock:
                detections = list(self.latest_detections)
                in_zone_count = self.last_in_zone_count

            output = frame.copy()

            if NIGHT_MODE:
                output = cv2.convertScaleAbs(output, alpha=1.15, beta=25)

            draw_zone(output, self.cam_id)

            if DETECTION_ENABLED and zone_ready(self.cam_id):
                for detection in detections:
                    x1, y1 = detection["x1"], detection["y1"]
                    x2, y2 = detection["x2"], detection["y2"]
                    label = f'{detection["class_name"].upper()} {detection["confidence"] * 100:.0f}%'
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(output, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.putText(output, f"CAM-{int(self.cam_id):02d}: {CAMERA_NAMES[self.cam_id]}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (79, 195, 247), 2)

            if not zone_ready(self.cam_id):
                status = "NO RESTRICTED ZONE - DETECTION OFF"
                status_color = (160, 160, 160)
            elif not DETECTION_ENABLED:
                status = "DETECTION OFF"
                status_color = (160, 160, 160)
            else:
                status = f"DETECTION ON - ZONE ACTIVE - {in_zone_count} OBJECT(S)"
                status_color = (0, 255, 0)

            cv2.putText(output, status, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2)
            self.publish_frame(output)

            frame_interval = 1.0 / max(self.fps, 1.0)
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.monotonic()

        if cap is not None:
            cap.release()

    def detection_loop(self):
        last_detection = 0.0
        last_debug = 0.0

        while self.running:
            now = time.monotonic()

            if not DETECTION_ENABLED or not zone_ready(self.cam_id):
                with self.detection_lock:
                    self.latest_detections = []
                    self.last_in_zone_count = 0
                    self.last_raw_count = 0
                    self.last_inference_ok = False
                time.sleep(0.05)
                continue

            if now - last_detection < DETECTION_INTERVAL:
                time.sleep(0.01)
                continue

            with self.raw_lock:
                frame = None if self.latest_raw is None else self.latest_raw.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            started = time.monotonic()
            detections, raw_count = detect_in_restricted_zone(frame, self.cam_id)
            elapsed = time.monotonic() - started

            self.last_inference_time = elapsed
            self.last_raw_count = raw_count
            self.last_error = ""
            self.last_inference_ok = True

            if not zone_ready(self.cam_id) or not DETECTION_ENABLED:
                detections = []

            with self.detection_lock:
                self.latest_detections = detections
                self.last_in_zone_count = len(detections)

            if detections:
                alert_frame = frame.copy()
                draw_zone(alert_frame, self.cam_id)
                for detection in detections:
                    cv2.rectangle(alert_frame, (detection["x1"], detection["y1"]), (detection["x2"], detection["y2"]), (0, 0, 255), 2)
                    create_alert(self.cam_id, detection["class_name"], detection["confidence"], alert_frame)

            if time.monotonic() - last_debug >= 3.0:
                print(
                    f"CAM-{int(self.cam_id):02d}: "
                    f"ZONE=ON | YOLO=OK | raw_objects={raw_count} | "
                    f"objects_in_zone={len(detections)} | "
                    f"inference={elapsed:.2f}s"
                )
                last_debug = time.monotonic()

            last_detection = time.monotonic()

    def get_frame(self, old_sequence):
        with self.condition:
            if self.sequence == old_sequence:
                self.condition.wait(timeout=1.0)
            return self.latest_jpeg, self.sequence


workers = {}
for camera_id in ["1", "2"]:
    workers[camera_id] = CameraWorker(camera_id, CAMERA_SOURCES[camera_id])

print("=" * 60)
print("SMART SENTINELS STARTED")
print("YOLO model:", MODEL_PATH)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("Inference size:", INFERENCE_SIZE)
print("Detection mode: FULL-FRAME YOLO + STRICT ZONE FILTER")
for cam_id, source in CAMERA_SOURCES.items():
    print(f"CAM-{int(cam_id):02d}: {source if source else 'VIDEO NOT FOUND'}")
print("=" * 60)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in workers:
        return "Invalid camera", 404

    worker = workers[cam_id]

    def generate():
        last_sequence = -1
        while True:
            jpeg, sequence = worker.get_frame(last_sequence)
            if jpeg is None or sequence == last_sequence:
                continue
            last_sequence = sequence
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/alerts")
def api_alerts():
    with alert_lock:
        return jsonify(list(recent_alerts))


@app.route("/api/sources")
def api_sources():
    return jsonify({cam_id: (os.path.basename(path) if path else None) for cam_id, path in CAMERA_SOURCES.items()})


@app.route("/api/zone_status")
def api_zone_status():
    with zone_lock:
        return jsonify({cam_id: {"points": len(points), "ready": len(points) >= 3} for cam_id, points in ZONE_POINTS.items()})


@app.route("/api/diagnostics")
def api_diagnostics():
    data = {}
    for cam_id, worker in workers.items():
        with worker.detection_lock:
            data[cam_id] = {
                "zone_ready": zone_ready(cam_id),
                "detection_enabled": DETECTION_ENABLED,
                "yolo_ok": worker.last_inference_ok,
                "raw_objects": worker.last_raw_count,
                "objects_in_zone": worker.last_in_zone_count,
                "inference_seconds": round(worker.last_inference_time, 3),
                "error": worker.last_error
            }
    return jsonify(data)


@app.route("/api/add_point", methods=["POST"])
def api_add_point():
    data = request.get_json(silent=True) or {}
    cam_id = str(data.get("cam_id", ""))
    if cam_id not in ZONE_POINTS:
        return jsonify({"status": "error", "message": "Invalid camera"}), 400

    try:
        x = float(data["x_percent"])
        y = float(data["y_percent"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid point"}), 400

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    with zone_lock:
        if len(ZONE_POINTS[cam_id]) >= 20:
            return jsonify({"status": "error", "message": "Maximum 20 points"}), 400
        ZONE_POINTS[cam_id].append((x, y))
        count = len(ZONE_POINTS[cam_id])

    return jsonify({"status": "success", "points": count, "ready": count >= 3})


@app.route("/api/clear_zone", methods=["POST"])
def api_clear_zone():
    data = request.get_json(silent=True) or {}
    cam_id = str(data.get("cam_id", ""))
    if cam_id not in ZONE_POINTS:
        return jsonify({"status": "error"}), 400

    with zone_lock:
        ZONE_POINTS[cam_id] = []

    worker = workers[cam_id]
    with worker.detection_lock:
        worker.latest_detections = []
        worker.last_in_zone_count = 0
        worker.last_raw_count = 0

    clear_camera_alerts(cam_id)
    return jsonify({"status": "success", "ready": False})


@app.route("/api/toggle_detection", methods=["POST"])
def api_toggle_detection():
    global DETECTION_ENABLED
    DETECTION_ENABLED = not DETECTION_ENABLED
    if not DETECTION_ENABLED:
        for worker in workers.values():
            with worker.detection_lock:
                worker.latest_detections = []
                worker.last_in_zone_count = 0
    return jsonify({"detection_enabled": DETECTION_ENABLED})


@app.route("/api/detection_status")
def api_detection_status():
    return jsonify({"detection_enabled": DETECTION_ENABLED})


@app.route("/api/toggle_alarm", methods=["POST"])
def api_toggle_alarm():
    global ALARM_ENABLED
    ALARM_ENABLED = not ALARM_ENABLED
    return jsonify({"alarm_enabled": ALARM_ENABLED})


@app.route("/api/toggle_night", methods=["POST"])
def api_toggle_night():
    global NIGHT_MODE
    NIGHT_MODE = not NIGHT_MODE
    return jsonify({"night_mode": NIGHT_MODE})


if __name__ == "__main__":
    print("\nSMART SENTINELS")
    print("Open: http://127.0.0.1:5000")
    print("FULL-FRAME YOLO + RESTRICTED-ZONE FILTER")
    print("Detection is OFF automatically when no zone exists.")
    app.run(debug=True, threaded=True, use_reloader=False, host="127.0.0.1", port=5000)
