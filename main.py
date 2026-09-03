import cv2
import socket
import json
from collections import deque
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat

# UDP config
UDP_IP = "127.0.0.1"
UDP_PORT = 5052
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Tasks API setup
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ===== SMOOTHING CONFIGURATION =====
SMOOTHING_METHOD = "exponential"
ALPHA = 0.3

# Moving Average settings
BUFFER_SIZE = 5

# Kalman-like simple filter
PROCESS_NOISE = 0.01
MEASUREMENT_NOISE = 0.1

# Initialize buffers for 42 landmarks (21 per hand)
if SMOOTHING_METHOD == "moving_average":
    landmark_buffers = [[deque(maxlen=BUFFER_SIZE) for _ in range(3)] for _ in range(42)]
elif SMOOTHING_METHOD == "exponential":
    previous_landmarks = None
elif SMOOTHING_METHOD == "kalman_simple":
    kalman_estimates = None
    kalman_uncertainties = None

def smooth_moving_average(landmark_list):
    smoothed = []
    for i, lm in enumerate(landmark_list):
        landmark_buffers[i][0].append(lm["x"])
        landmark_buffers[i][1].append(lm["y"])
        landmark_buffers[i][2].append(lm["z"])
        smoothed.append({
            "x": float(np.mean(landmark_buffers[i][0])),
            "y": float(np.mean(landmark_buffers[i][1])),
            "z": float(np.mean(landmark_buffers[i][2]))
        })
    return smoothed

def smooth_exponential(landmark_list):
    global previous_landmarks
    if previous_landmarks is None or len(previous_landmarks) != len(landmark_list):
        previous_landmarks = landmark_list
        return landmark_list
    smoothed = []
    for i, lm in enumerate(landmark_list):
        smoothed.append({
            "x": ALPHA * lm["x"] + (1 - ALPHA) * previous_landmarks[i]["x"],
            "y": ALPHA * lm["y"] + (1 - ALPHA) * previous_landmarks[i]["y"],
            "z": ALPHA * lm["z"] + (1 - ALPHA) * previous_landmarks[i]["z"]
        })
    previous_landmarks = smoothed
    return smoothed

def smooth_kalman_simple(landmark_list):
    global kalman_estimates, kalman_uncertainties
    if kalman_estimates is None or len(kalman_estimates) != len(landmark_list):
        kalman_estimates = landmark_list
        kalman_uncertainties = [{"x": 1.0, "y": 1.0, "z": 1.0} for _ in range(len(landmark_list))]
        return landmark_list
    smoothed = []
    for i, lm in enumerate(landmark_list):
        k_x = kalman_uncertainties[i]["x"] / (kalman_uncertainties[i]["x"] + MEASUREMENT_NOISE)
        k_y = kalman_uncertainties[i]["y"] / (kalman_uncertainties[i]["y"] + MEASUREMENT_NOISE)
        k_z = kalman_uncertainties[i]["z"] / (kalman_uncertainties[i]["z"] + MEASUREMENT_NOISE)
        est_x = kalman_estimates[i]["x"] + k_x * (lm["x"] - kalman_estimates[i]["x"])
        est_y = kalman_estimates[i]["y"] + k_y * (lm["y"] - kalman_estimates[i]["y"])
        est_z = kalman_estimates[i]["z"] + k_z * (lm["z"] - kalman_estimates[i]["z"])
        kalman_uncertainties[i]["x"] = (1 - k_x) * kalman_uncertainties[i]["x"] + PROCESS_NOISE
        kalman_uncertainties[i]["y"] = (1 - k_y) * kalman_uncertainties[i]["y"] + PROCESS_NOISE
        kalman_uncertainties[i]["z"] = (1 - k_z) * kalman_uncertainties[i]["z"] + PROCESS_NOISE
        smoothed.append({"x": est_x, "y": est_y, "z": est_z})
        kalman_estimates[i] = smoothed[i]
    return smoothed

def apply_smoothing(landmark_list):
    if SMOOTHING_METHOD == "moving_average":
        return smooth_moving_average(landmark_list)
    elif SMOOTHING_METHOD == "exponential":
        return smooth_exponential(landmark_list)
    elif SMOOTHING_METHOD == "kalman_simple":
        return smooth_kalman_simple(landmark_list)
    else:
        return landmark_list

while True:
    success, img = cap.read()
    if not success:
        continue

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)

    # Detect hands
    result = detector.detect(mp_image)

    all_landmarks = []
    right_hand_detected = False
    left_hand_detected = False

    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand_label = handedness[0].category_name  # "Left" or "Right"

            hand_landmark_list = []
            for lm in hand_landmarks:
                hand_landmark_list.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

            if hand_label == "Right":
                if not right_hand_detected:
                    all_landmarks = hand_landmark_list + all_landmarks
                    right_hand_detected = True
            else:
                if not left_hand_detected:
                    all_landmarks = all_landmarks + hand_landmark_list
                    left_hand_detected = True

            # Draw landmarks on image
            for lm in hand_landmarks:
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

    if all_landmarks:
        smoothed_landmarks = apply_smoothing(all_landmarks)
        packet = {
            "landmarks": smoothed_landmarks,
            "right_hand_detected": right_hand_detected,
            "left_hand_detected": left_hand_detected
        }
        json_string = json.dumps(packet)
        print(f"Hands: R={right_hand_detected} L={left_hand_detected}, Landmarks: {len(smoothed_landmarks)}")
        sock.sendto(json_string.encode('utf-8'), (UDP_IP, UDP_PORT))

    cv2.putText(img, f"Smoothing: {SMOOTHING_METHOD}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Right: {right_hand_detected}  Left: {left_hand_detected}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()