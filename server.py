from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import requests, numpy as np, cv2, threading, time, math, json, os, traceback

# ==================================================
# CONFIG
# ==================================================
PHONE_URL = "http://192.168.225.212:8080/shot.jpg"
MODEL_PATH = "best.pt"

# deduplication
CENTER_DISTANCE_THRESHOLD = 80  # pixels (conservative)
ACTIVE_TTL = 3.0  # seconds a crack stays active if not seen
YOLO_INTERVAL = 1.0  # seconds between inference

# persistent storage file
PERSIST_FILE = "cumulative_stats.json"

# load YOLO
model = YOLO(MODEL_PATH)

# Flask
app = Flask(__name__)
CORS(app)


# ==================================================
# PERSISTENT STORAGE
# ==================================================
def load_persistent_stats():
    if not os.path.exists(PERSIST_FILE):
        return {
            "total_cracks": 0,
            "total_severe": 0,
            "total_moderate": 0,
            "total_minor": 0,
        }
    try:
        with open(PERSIST_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "total_cracks": 0,
            "total_severe": 0,
            "total_moderate": 0,
            "total_minor": 0,
        }


def save_persistent_stats():
    with open(PERSIST_FILE, "w") as f:
        json.dump(persistent_stats, f)


persistent_stats = load_persistent_stats()


def update_totals(sev):
    persistent_stats["total_cracks"] += 1
    if sev == "Severe":
        persistent_stats["total_severe"] += 1
    elif sev == "Moderate":
        persistent_stats["total_moderate"] += 1
    else:
        persistent_stats["total_minor"] += 1

    save_persistent_stats()


# ==================================================
# SHARED STATE (current session)
# ==================================================
latest_boxes = []
latest_masks = []
latest_count = 0
latest_severity = "None"

active_cracks = {}  # id: info
next_crack_id = 1

lock = threading.Lock()


# ==================================================
# HELPERS
# ==================================================
def get_frame():
    try:
        data = requests.get(PHONE_URL, timeout=1).content
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except:
        return None


def classify_severity(area):
    if area < 1500:
        return "Minor"
    elif area < 5000:
        return "Moderate"
    else:
        return "Severe"


def mask_to_bbox(mask, frame_shape):
    h, w = frame_shape[:2]
    mask_resized = cv2.resize(mask, (w, h))
    bin_mask = (mask_resized > 0.5).astype(np.uint8)

    ys, xs = np.where(bin_mask == 1)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    wbox = max(1, x2 - x1)
    hbox = max(1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    area = int(bin_mask.sum())

    return {
        "x": int(x1),
        "y": int(y1),
        "w": int(wbox),
        "h": int(hbox),
        "cx": int(cx),
        "cy": int(cy),
        "area": area,
        "mask": bin_mask,
    }


def center_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def clean_old_cracks(now):
    remove_ids = []
    for cid, info in active_cracks.items():
        if now - info["last_seen"] > ACTIVE_TTL:
            remove_ids.append(cid)
    for cid in remove_ids:
        del active_cracks[cid]


# ==================================================
# YOLO THREAD (segmentation + tracking + persistence)
# ==================================================
def yolo_thread():
    global latest_boxes, latest_masks, latest_count, latest_severity
    global next_crack_id, active_cracks

    print("YOLO THREAD STARTED ✓")

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        try:
            results = model(frame, conf=0.3, show=False)
        except:
            traceback.print_exc()
            time.sleep(YOLO_INTERVAL)
            continue

        r = results[0]
        masks = r.masks.data.cpu().numpy() if r.masks is not None else []

        now = time.time()

        # convert masks → bbox + center
        detections = []
        for mask in masks:
            info = mask_to_bbox(mask, frame.shape)
            if info is None:
                continue
            sev = classify_severity(info["area"])
            info["severity"] = sev
            detections.append(info)

        with lock:
            clean_old_cracks(now)

            for det in detections:
                dc = (det["cx"], det["cy"])
                matched_id = None
                best_dist = 1e9

                for cid, info in active_cracks.items():
                    dist = center_distance(dc, info["center"])
                    if dist < best_dist:
                        best_dist = dist
                        best_id = cid

                if best_dist <= CENTER_DISTANCE_THRESHOLD:
                    matched_id = best_id

                if matched_id is not None:
                    # update existing
                    info = active_cracks[matched_id]
                    info["center"] = dc
                    info["bbox"] = (det["x"], det["y"], det["w"], det["h"])
                    info["mask"] = det["mask"]
                    info["last_seen"] = now

                    # severity: keep worst
                    order = {"Minor": 1, "Moderate": 2, "Severe": 3}
                    if order[det["severity"]] > order[info["severity"]]:
                        info["severity"] = det["severity"]

                else:
                    # NEW CRACK
                    cid = next_crack_id
                    next_crack_id += 1

                    active_cracks[cid] = {
                        "center": dc,
                        "bbox": (det["x"], det["y"], det["w"], det["h"]),
                        "mask": det["mask"],
                        "severity": det["severity"],
                        "last_seen": now,
                    }

                    # persistent update
                    update_totals(det["severity"])

            # update latest for UI stream
            boxes = []
            masks_out = []
            worst_sev = "None"

            for cid, info in active_cracks.items():
                x, y, w, h = info["bbox"]
                sev = info["severity"]
                boxes.append(
                    {"id": cid, "x": x, "y": y, "w": w, "h": h, "severity": sev}
                )
                masks_out.append(info["mask"])

                if sev == "Severe":
                    worst_sev = "Severe"
                elif sev == "Moderate" and worst_sev != "Severe":
                    worst_sev = "Moderate"
                elif sev == "Minor" and worst_sev not in ["Moderate", "Severe"]:
                    worst_sev = "Minor"

            latest_boxes = boxes
            latest_masks = masks_out
            latest_count = len(boxes)
            latest_severity = worst_sev

        time.sleep(YOLO_INTERVAL)


# start thread
threading.Thread(target=yolo_thread, daemon=True).start()


# ==================================================
# STREAM LOOP
# ==================================================
def stream():
    global latest_boxes, latest_masks

    while True:
        frame = get_frame()
        if frame is None:
            continue

        with lock:
            for idx, b in enumerate(latest_boxes):
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                sev = b["severity"]

                # severity color
                if sev == "Severe":
                    color = (0, 0, 255)
                elif sev == "Moderate":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                # bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, f"{sev}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

                # segmentation overlay
                if idx < len(latest_masks) and latest_masks[idx] is not None:
                    mask_bin = latest_masks[idx]
                    mask_resized = cv2.resize(
                        mask_bin.astype(np.uint8), (frame.shape[1], frame.shape[0])
                    )
                    colored = np.zeros_like(frame)
                    colored[:] = color
                    alpha = 0.28
                    frame = np.where(
                        mask_resized[..., None] == 1,
                        cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0),
                        frame,
                    )

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


# ==================================================
# ROUTES
# ==================================================
@app.route("/video")
def video_feed():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/info")
def info():
    with lock:
        return jsonify(
            {
                "current_count": latest_count,
                "current_severity": latest_severity,
                "total_cracks": persistent_stats["total_cracks"],
                "total_severe": persistent_stats["total_severe"],
                "total_moderate": persistent_stats["total_moderate"],
                "total_minor": persistent_stats["total_minor"],
            }
        )


# ==================================================
# START
# ==================================================
if __name__ == "__main__":
    print("Server running on port 5000")
    app.run("0.0.0.0", 5000, threaded=True)
