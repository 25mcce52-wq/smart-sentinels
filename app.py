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


# ============================================================
# CAMERA
# ============================================================

CAMERA_ID = "2"
CAMERA_NAME = "TEST"
CAMERA_FILE = os.path.join(BASE_DIR, "cam2.mp4")


# ============================================================
# YOLO
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

model = YOLO(MODEL_PATH)

model_lock = threading.Lock()

TARGET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# Lower settings for Render free instance
CONFIDENCE_THRESHOLD = 0.20
INFERENCE_SIZE = 320
DETECTION_INTERVAL = 0.35
ALERT_COOLDOWN = 3.0


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DETECTION_ENABLED = True
ALARM_ENABLED = False
NIGHT_MODE = False


# ============================================================
# RESTRICTED ZONE
# ============================================================

ZONE_POINTS = []

zone_lock = threading.Lock()


# ============================================================
# ALERTS
# ============================================================

recent_alerts = []

alert_lock = threading.Lock()

last_alert_time = {}


# ============================================================
# ZONE FUNCTIONS
# ============================================================

def zone_ready():
    with zone_lock:
        return len(ZONE_POINTS) >= 3


def get_zone_points():
    with zone_lock:
        return list(ZONE_POINTS)


def clear_camera_alerts():
    with alert_lock:
        recent_alerts.clear()
        last_alert_time.clear()


# ============================================================
# ALARM
# ============================================================

def trigger_alarm():

    if not ALARM_ENABLED:
        return

    if winsound is None:
        return

    try:
        winsound.Beep(2500, 700)
    except Exception:
        pass


# ============================================================
# SAVE EVIDENCE
# ============================================================

def save_evidence(frame, class_name):

    try:

        now = datetime.now()

        filename = (
            f"TEST_{class_name}_"
            f"{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        )

        path = os.path.join(EVIDENCE_DIR, filename)

        cv2.imwrite(path, frame)

        return filename

    except Exception as error:

        print("Evidence save error:", error)

        return None


# ============================================================
# CREATE ALERT
# ============================================================

def create_alert(class_name, confidence, frame):

    if not zone_ready():
        return

    if not DETECTION_ENABLED:
        return

    now = time.monotonic()

    key = class_name

    with alert_lock:

        if now - last_alert_time.get(key, 0) < ALERT_COOLDOWN:
            return

        last_alert_time[key] = now

        evidence = save_evidence(frame, class_name)

        recent_alerts.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "camera": "CAM-02",
                "event": f"{class_name.upper()} entered restricted zone",
                "severity": "CRITICAL",
                "confidence": round(confidence * 100, 1),
                "evidence": evidence
            }
        )

        if len(recent_alerts) > 50:
            recent_alerts.pop()

    threading.Thread(
        target=trigger_alarm,
        daemon=True
    ).start()


# ============================================================
# ZONE GEOMETRY
# ============================================================

def zone_polygon_for_frame(frame):

    points = get_zone_points()

    if len(points) < 3:
        return None

    height, width = frame.shape[:2]

    pixel_points = [
        (
            round(x * (width - 1)),
            round(y * (height - 1))
        )
        for x, y in points
    ]

    return np.array(
        pixel_points,
        dtype=np.int32
    ).reshape((-1, 1, 2))


def draw_zone(frame):

    points = get_zone_points()

    if not points:
        return

    height, width = frame.shape[:2]

    pixel_points = [
        (
            round(x * (width - 1)),
            round(y * (height - 1))
        )
        for x, y in points
    ]

    pts = np.array(
        pixel_points,
        dtype=np.int32
    ).reshape((-1, 1, 2))

    if len(pixel_points) >= 3:

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 0, 255)
        )

        cv2.addWeighted(
            overlay,
            0.12,
            frame,
            0.88,
            0,
            frame
        )

        cv2.polylines(
            frame,
            [pts],
            True,
            (0, 0, 255),
            3
        )

    else:

        cv2.polylines(
            frame,
            [pts],
            False,
            (0, 0, 255),
            3
        )

        for point in pixel_points:

            cv2.circle(
                frame,
                point,
                6,
                (0, 0, 255),
                -1
            )


# ============================================================
# BOUNDING BOX / ZONE CHECK
# ============================================================

def bbox_touches_zone(
    polygon,
    x1,
    y1,
    x2,
    y2
):

    if polygon is None:
        return False

    center = (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )

    bottom = (
        (x1 + x2) // 2,
        y2
    )

    corners = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2)
    ]

    for point in [center, bottom] + corners:

        if cv2.pointPolygonTest(
            polygon,
            point,
            False
        ) >= 0:

            return True

    return False


# ============================================================
# YOLO CLASS NAME
# ============================================================

def model_class_name(class_id):

    names = model.names

    if isinstance(names, dict):

        return str(
            names.get(
                class_id,
                class_id
            )
        ).lower()

    if 0 <= class_id < len(names):

        return str(
            names[class_id]
        ).lower()

    return str(class_id).lower()


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_objects(frame):

    if not DETECTION_ENABLED:
        return [], 0

    if not zone_ready():
        return [], 0

    polygon = zone_polygon_for_frame(frame)

    if polygon is None:
        return [], 0

    detections = []

    raw_count = 0

    try:

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

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model_class_name(
                        class_id
                    )

                    if class_name not in TARGET_CLASSES:
                        continue

                    x1, y1, x2, y2 = [
                        int(v)
                        for v in box.xyxy[0].tolist()
                    ]

                    x1 = max(
                        0,
                        min(
                            frame.shape[1] - 1,
                            x1
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            frame.shape[0] - 1,
                            y1
                        )
                    )

                    x2 = max(
                        0,
                        min(
                            frame.shape[1] - 1,
                            x2
                        )
                    )

                    y2 = max(
                        0,
                        min(
                            frame.shape[0] - 1,
                            y2
                        )
                    )

                    if x2 <= x1 or y2 <= y1:
                        continue

                    if not bbox_touches_zone(
                        polygon,
                        x1,
                        y1,
                        x2,
                        y2
                    ):
                        continue

                    detections.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "class_name": class_name,
                            "confidence": confidence
                        }
                    )

                except Exception as box_error:

                    print(
                        "Detection box error:",
                        box_error
                    )

    except Exception as error:

        print(
            "YOLO error:",
            type(error).__name__,
            error
        )

    return detections, raw_count


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker:

    def __init__(self, source_path):

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

        self.fps = 15.0

        self.condition = threading.Condition(
            self.output_lock
        )

        threading.Thread(
            target=self.capture_loop,
            daemon=True
        ).start()

        threading.Thread(
            target=self.detection_loop,
            daemon=True
        ).start()


    def create_status_frame(self, message):

        frame = np.zeros(
            (360, 640, 3),
            dtype=np.uint8
        )

        cv2.putText(
            frame,
            "TEST",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (79, 195, 247),
            3
        )

        cv2.putText(
            frame,
            message,
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (190, 190, 190),
            2
        )

        return frame


    def open_video(self):

        if not self.source_path:
            return None

        print(
            "TEST camera source:",
            self.source_path
        )

        cap = cv2.VideoCapture(
            self.source_path
        )

        if not cap.isOpened():

            cap.release()

            return None

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0 or fps > 120:

            fps = 15.0

        self.fps = min(
            fps,
            15.0
        )

        return cap


    def publish_frame(self, frame):

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                70
            ]
        )

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

                    self.publish_frame(
                        self.create_status_frame(
                            "Camera video file not found"
                        )
                    )

                    time.sleep(2)

                    continue

                next_frame_time = time.monotonic()


            ok, frame = cap.read()

            if not ok:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                next_frame_time = time.monotonic()

                continue


            with self.raw_lock:

                self.latest_raw = frame.copy()


            with self.detection_lock:

                detections = list(
                    self.latest_detections
                )

                in_zone_count = (
                    self.last_in_zone_count
                )


            output = frame.copy()


            # Optional night filter
            if NIGHT_MODE:

                output = cv2.convertScaleAbs(
                    output,
                    alpha=1.15,
                    beta=25
                )


            draw_zone(output)


            if (
                DETECTION_ENABLED
                and zone_ready()
            ):

                for detection in detections:

                    x1 = detection["x1"]
                    y1 = detection["y1"]
                    x2 = detection["x2"]
                    y2 = detection["y2"]

                    label = (
                        f'{detection["class_name"].upper()} '
                        f'{detection["confidence"] * 100:.0f}%'
                    )

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        output,
                        label,
                        (
                            x1,
                            max(22, y1 - 7)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )


            cv2.putText(
                output,
                "CAM-02: TEST",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (79, 195, 247),
                2
            )


            if not zone_ready():

                status = (
                    "NO RESTRICTED ZONE - "
                    "DETECTION OFF"
                )

                status_color = (
                    160,
                    160,
                    160
                )

            elif not DETECTION_ENABLED:

                status = "DETECTION OFF"

                status_color = (
                    160,
                    160,
                    160
                )

            else:

                status = (
                    f"DETECTION ON - "
                    f"ZONE ACTIVE - "
                    f"{in_zone_count} OBJECT(S)"
                )

                status_color = (
                    0,
                    255,
                    0
                )


            cv2.putText(
                output,
                status,
                (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                status_color,
                2
            )


            self.publish_frame(output)


            frame_interval = (
                1.0 /
                max(self.fps, 1.0)
            )

            next_frame_time += frame_interval

            sleep_time = (
                next_frame_time -
                time.monotonic()
            )

            if sleep_time > 0:

                time.sleep(
                    sleep_time
                )

            else:

                next_frame_time = (
                    time.monotonic()
                )


        if cap is not None:

            cap.release()


    def detection_loop(self):

        last_detection = 0.0

        last_debug = 0.0

        while self.running:

            now = time.monotonic()


            if (
                not DETECTION_ENABLED
                or not zone_ready()
            ):

                with self.detection_lock:

                    self.latest_detections = []

                    self.last_in_zone_count = 0

                    self.last_raw_count = 0

                    self.last_inference_ok = False

                time.sleep(0.05)

                continue


            if (
                now - last_detection
                < DETECTION_INTERVAL
            ):

                time.sleep(0.01)

                continue


            with self.raw_lock:

                frame = (
                    None
                    if self.latest_raw is None
                    else self.latest_raw.copy()
                )


            if frame is None:

                time.sleep(0.03)

                continue


            started = time.monotonic()


            detections, raw_count = (
                detect_objects(frame)
            )


            elapsed = (
                time.monotonic()
                - started
            )


            self.last_inference_time = elapsed

            self.last_raw_count = raw_count

            self.last_error = ""

            self.last_inference_ok = True


            with self.detection_lock:

                self.latest_detections = (
                    detections
                )

                self.last_in_zone_count = (
                    len(detections)
                )


            if detections:

                alert_frame = frame.copy()

                draw_zone(alert_frame)

                for detection in detections:

                    cv2.rectangle(
                        alert_frame,
                        (
                            detection["x1"],
                            detection["y1"]
                        ),
                        (
                            detection["x2"],
                            detection["y2"]
                        ),
                        (0, 0, 255),
                        2
                    )

                    create_alert(
                        detection["class_name"],
                        detection["confidence"],
                        alert_frame
                    )


            if (
                time.monotonic()
                - last_debug
                >= 3.0
            ):

                print(
                    "TEST: "
                    f"ZONE=ON | "
                    f"YOLO=OK | "
                    f"raw_objects={raw_count} | "
                    f"objects_in_zone={len(detections)} | "
                    f"inference={elapsed:.2f}s"
                )

                last_debug = time.monotonic()


            last_detection = (
                time.monotonic()
            )


    def get_frame(self, old_sequence):

        with self.condition:

            if self.sequence == old_sequence:

                self.condition.wait(
                    timeout=1.0
                )

            return (
                self.latest_jpeg,
                self.sequence
            )


# ============================================================
# START CAMERA WORKER
# ============================================================

worker = CameraWorker(
    CAMERA_FILE
)


print("=" * 60)

print("SMART SENTINELS STARTED")

print("CAMERA: CAM-02")

print("NAME: TEST")

print("VIDEO:", CAMERA_FILE)

print("YOLO:", MODEL_PATH)

print(
    "Confidence threshold:",
    CONFIDENCE_THRESHOLD
)

print(
    "Inference size:",
    INFERENCE_SIZE
)

print("Only one camera worker enabled")

print("=" * 60)


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    def generate():

        last_sequence = -1

        while True:

            jpeg, sequence = (
                worker.get_frame(
                    last_sequence
                )
            )

            if (
                jpeg is None
                or sequence == last_sequence
            ):
                continue

            last_sequence = sequence

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg +
                b"\r\n"
            )


    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# ALERT API
# ============================================================

@app.route("/api/alerts")
def api_alerts():

    with alert_lock:

        return jsonify(
            list(recent_alerts)
        )


# ============================================================
# SOURCE API
# ============================================================

@app.route("/api/sources")
def api_sources():

    return jsonify(
        {
            "2": "cam2.mp4"
        }
    )


# ============================================================
# ZONE STATUS
# ============================================================

@app.route("/api/zone_status")
def api_zone_status():

    with zone_lock:

        return jsonify(
            {
                "2": {
                    "points": len(
                        ZONE_POINTS
                    ),
                    "ready": len(
                        ZONE_POINTS
                    ) >= 3
                }
            }
        )


# ============================================================
# DIAGNOSTICS
# ============================================================

@app.route("/api/diagnostics")
def api_diagnostics():

    with worker.detection_lock:

        return jsonify(
            {
                "2": {
                    "zone_ready": zone_ready(),
                    "detection_enabled":
                        DETECTION_ENABLED,
                    "yolo_ok":
                        worker.last_inference_ok,
                    "raw_objects":
                        worker.last_raw_count,
                    "objects_in_zone":
                        worker.last_in_zone_count,
                    "inference_seconds":
                        round(
                            worker.last_inference_time,
                            3
                        ),
                    "error":
                        worker.last_error
                }
            }
        )


# ============================================================
# ADD ZONE POINT
# ============================================================

@app.route(
    "/api/add_point",
    methods=["POST"]
)
def api_add_point():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    cam_id = str(
        data.get(
            "cam_id",
            ""
        )
    )


    if cam_id != CAMERA_ID:

        return jsonify(
            {
                "status": "error",
                "message":
                    "Invalid camera"
            }
        ), 400


    try:

        x = float(
            data["x_percent"]
        )

        y = float(
            data["y_percent"]
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        return jsonify(
            {
                "status": "error",
                "message":
                    "Invalid point"
            }
        ), 400


    x = max(
        0.0,
        min(1.0, x)
    )

    y = max(
        0.0,
        min(1.0, y)
    )


    with zone_lock:

        if len(ZONE_POINTS) >= 20:

            return jsonify(
                {
                    "status": "error",
                    "message":
                        "Maximum 20 points"
                }
            ), 400


        ZONE_POINTS.append(
            (x, y)
        )

        count = len(
            ZONE_POINTS
        )


    return jsonify(
        {
            "status": "success",
            "points": count,
            "ready": count >= 3
        }
    )


# ============================================================
# CLEAR ZONE
# ============================================================

@app.route(
    "/api/clear_zone",
    methods=["POST"]
)
def api_clear_zone():

    with zone_lock:

        ZONE_POINTS.clear()


    with worker.detection_lock:

        worker.latest_detections = []

        worker.last_in_zone_count = 0

        worker.last_raw_count = 0


    clear_camera_alerts()


    return jsonify(
        {
            "status": "success",
            "ready": False
        }
    )


# ============================================================
# TOGGLE AI DETECTION
# ============================================================

@app.route(
    "/api/toggle_detection",
    methods=["POST"]
)
def api_toggle_detection():

    global DETECTION_ENABLED

    DETECTION_ENABLED = (
        not DETECTION_ENABLED
    )


    if not DETECTION_ENABLED:

        with worker.detection_lock:

            worker.latest_detections = []

            worker.last_in_zone_count = 0


    return jsonify(
        {
            "detection_enabled":
                DETECTION_ENABLED
        }
    )


@app.route(
    "/api/detection_status"
)
def api_detection_status():

    return jsonify(
        {
            "detection_enabled":
                DETECTION_ENABLED
        }
    )


# ============================================================
# TOGGLE ALARM
# ============================================================

@app.route(
    "/api/toggle_alarm",
    methods=["POST"]
)
def api_toggle_alarm():

    global ALARM_ENABLED

    ALARM_ENABLED = (
        not ALARM_ENABLED
    )


    return jsonify(
        {
            "alarm_enabled":
                ALARM_ENABLED
        }
    )


# ============================================================
# TOGGLE NIGHT FILTER
# ============================================================

@app.route(
    "/api/toggle_night",
    methods=["POST"]
)
def api_toggle_night():

    global NIGHT_MODE

    NIGHT_MODE = (
        not NIGHT_MODE
    )


    return jsonify(
        {
            "night_mode":
                NIGHT_MODE
        }
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\nSMART SENTINELS"
    )

    print(
        "CAM-02: TEST"
    )

    print(
        "VIDEO: cam2.mp4"
    )

    print(
        "Only one camera enabled"
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    app.run(
        debug=True,
        threaded=True,
        use_reloader=False,
        host="127.0.0.1",
        port=5000
    )