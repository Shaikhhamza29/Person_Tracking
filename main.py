import cv2
import onnxruntime as ort
import numpy as np
import time
import sys
import os

# -------------------------
# RESOURCE PATH
# -------------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# -------------------------
# LOAD MODEL
# -------------------------
session = ort.InferenceSession(resource_path("yolov8n.onnx"))
input_name = session.get_inputs()[0].name

print("⚡ ONNX Model Loaded")

# -------------------------
# VIDEO SOURCES
# -------------------------
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video = cv2.VideoCapture(resource_path("2.mp4"))
import ctypes

user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

cv2.namedWindow("AI Display", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cam.set(3, 320)
cam.set(4, 240)

# -------------------------
# PIXELATE
# -------------------------
def pixelate(frame, scale=0.15):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# -------------------------
# PREPROCESS (FIXED 640)
# -------------------------
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))  # MUST be 640
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# DETECT PERSON (CORRECT)
# -------------------------
def detect_person(frame):
    input_data = preprocess(frame)
    outputs = session.run(None, {input_name: input_data})[0]

    outputs = np.squeeze(outputs)

    # Handle both shapes automatically
    if outputs.shape[0] == 84:
        outputs = outputs.T

    # Object confidence
    obj_conf = outputs[:, 4]

    # 🔥 DEBUG
    max_conf = np.max(obj_conf)
    print("Max obj conf:", max_conf)

    # 🔥 SIMPLE + RELIABLE DETECTION
    return max_conf > 0.25

# -------------------------
# CONTROL VARIABLES
# -------------------------
last_detected_time = 0
hold_time = 1.2

transition_alpha = 0.0
transition_speed = 0.05

frame_count = 0
detect_every = 4

person_detected = False

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret_cam, cam_frame = cam.read()
    frame_count += 1

    # DETECTION
    if ret_cam and frame_count % detect_every == 0:
        person_detected = detect_person(cam_frame)

        if person_detected:
            last_detected_time = time.time()

    # VIDEO
    ret_vid, vid_frame = video.read()

    if not ret_vid:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # TRANSITION
    target = 1.0 if (time.time() - last_detected_time < hold_time) else 0.0

    if transition_alpha < target:
        transition_alpha = min(transition_alpha + transition_speed, 1.0)
    else:
        transition_alpha = max(transition_alpha - transition_speed, 0.0)

    # PIXEL EFFECT
    if transition_alpha < 1:
        pixel_frame = pixelate(vid_frame)
    else:
        pixel_frame = vid_frame

    output = cv2.addWeighted(
        vid_frame, transition_alpha,
        pixel_frame, 1 - transition_alpha,
        0
    )

    # STATUS TEXT
    status = "PERSON DETECTED" if transition_alpha > 0.5 else "NO PERSON"
    color = (255, 255, 0) if transition_alpha > 0.5 else (0, 0, 255)

    cv2.putText(output, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2)

    output = cv2.resize(output, (screen_width, screen_height))
    cv2.imshow("AI Display", output)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# CLEANUP
cam.release()
video.release()
cv2.destroyAllWindows()
