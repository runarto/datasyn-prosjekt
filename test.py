# YOLOv8 + Tracking (SSH-safe, image folder based)

import os
from ultralytics import YOLO
import cv2
import numpy as np
from norfair import Detection, Tracker

# === CONFIG ===
image_folder =  "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/img1"
output_folder = "output/annotated_frames"
model_path = "runs/detect/train13/weights/best.pt"  # Your trained model
os.makedirs(output_folder, exist_ok=True)

# === Tracker Setup ===
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

# === Load YOLO model ===
model = YOLO(model_path)

# === Process images ===
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])

for fname in image_files:
    img_path = os.path.join(image_folder, fname)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[Warning] Failed to read {fname}")
        continue

    results = model(frame)[0]
    detections = []

    for r in results.boxes:
        cls = int(r.cls[0])
        if cls in [0, 1]:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(Detection(np.array([cx, cy])))
            label = "ball" if cls == 0 else "player"
            color = (0, 0, 255) if cls == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    tracked_objects = tracker.update(detections=detections)
    for t in tracked_objects:
        (cx, cy) = map(int, t.estimate[0])  # handle shape (1, 2)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
        cv2.putText(frame, f"ID {t.id}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    out_path = os.path.join(output_folder, fname)
    cv2.imwrite(out_path, frame)
    print(f"Saved annotated: {out_path}")

print("✅ All images processed and saved to:", output_folder)