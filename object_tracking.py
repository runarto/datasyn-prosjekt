from ultralytics import YOLO
import cv2
import numpy as np
import os

class ObjectTracking:
    def __init__(self, image_folder, ball_weights='runs/detect/train_ball/weights/best.pt', player_weights='runs/detect/train_player/weights/best.pt', auto_calibrate=False):
        self.image_folder = image_folder
        self.ball_model = YOLO(ball_weights)
        self.player_model = YOLO(player_weights)

        self.class_id = {
            'ball': 0,
            'player': 1
        }

        self.colors = {
            'ball': (0, 0, 255),     # Red
            'player': (0, 255, 0),   # Green
        }

        self.K = None
        if auto_calibrate:
            self.K = self.Calibrate()

    def Calibrate(self, max_images=20, randomize=False):
        checkerboard = (8, 6)

        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        image_files = sorted([
            os.path.join(self.image_folder, fname)
            for fname in os.listdir(self.image_folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if randomize:
            import random
            image_files = random.sample(image_files, min(max_images, len(image_files)))
        else:
            image_files = image_files[:max_images]

        for fname in image_files:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

            print(f"{fname} → corners found: {ret}")

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) < 3:
            raise ValueError("Not enough valid images found for calibration.")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.K = K

        print("Camera calibration successful!")
        print("Intrinsic matrix (K):\n", K)

        return K


    def detect(self, model, frame, label, conf):
        results = model.predict(frame, conf=conf)[0]
        boxes = results.boxes.xyxy
        confs = results.boxes.conf
        classes = [self.class_id[label]] * len(boxes)  # override class index

        return boxes, confs, classes

    def draw_boxes(self, frame, boxes, confs, classes, label_type):
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box.tolist())
            confidence = float(conf.item())
            label = label_type

            print(f"Detected: {label} with conf {confidence:.2f}")

            color = self.colors.get(label, (255, 255, 255))
            text = f"{label} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def track_object(self, frame):
        # Predict separately for ball and player
        ball_boxes, ball_confs, ball_classes = self.detect(self.ball_model, frame, 'ball', 0.10)
        player_boxes, player_confs, player_classes = self.detect(self.player_model, frame, 'player', 0.6)

        # --- Keep only top 22 players by confidence ---
        player_data = sorted(zip(player_boxes, player_confs, player_classes), key=lambda x: x[1], reverse=True)
        player_data = player_data[:22]  # Keep top 22
        player_boxes, player_confs, player_classes = zip(*player_data) if player_data else ([], [], [])

        # Combine predictions for output
        boxes = list(ball_boxes) + list(player_boxes)
        confs = list(ball_confs) + list(player_confs)
        classes = list(ball_classes) + list(player_classes)

        # Draw predictions
        self.draw_boxes(frame, ball_boxes, ball_confs, ball_classes, 'ball')
        self.draw_boxes(frame, player_boxes, player_confs, player_classes, 'player')

        # Prepare label data for downstream usage
        labels = []
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = 'ball' if cls == self.class_id['ball'] else 'player'
            labels.append({
                'label': label,
                'confidence': float(conf.item()),
                'bbox': [x1, y1, x2, y2]
            })

        cv2.imshow("Detections", frame)
        cv2.waitKey(1)

        return labels





