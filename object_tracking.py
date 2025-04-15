from ultralytics import YOLO
import cv2
import numpy as np
import os
import supervision as sv

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
            'ball': sv.Color.BLUE,
            'player': sv.Color.RED,
        }

        self.K = None
        if auto_calibrate:
            self.K = self.Calibrate()

        self.ball_annotator = sv.EllipseAnnotator(color=self.colors['ball'], thickness=3)
        self.player_annotator = sv.EllipseAnnotator(color=self.colors['player'], thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=3, color=sv.Color.BLACK
        )

    def Calibrate(self, max_images=20, randomize=False):
        return self.K  # Placeholder for future calibration logic

    def detect(self, model, frame, label, conf):
        results = model.predict(frame, conf=conf)[0]
        boxes = results.boxes.xyxy
        confs = results.boxes.conf
        classes = [self.class_id[label]] * len(boxes)  # override class index

        xyxy = np.array([box.tolist() for box in boxes])
        class_id = np.array(classes)
        confidence = np.array([c.item() for c in confs])

        detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)
        return detections

    def draw_detections(self, frame, detections):
        if len(detections) == 0:
            return frame

        if detections.class_id is not None:
            mask_player = detections.class_id == self.class_id['player']
            mask_ball = detections.class_id == self.class_id['ball']

            if any(mask_player):
                frame = self.player_annotator.annotate(frame, detections[mask_player])
            if any(mask_ball):
                frame = self.ball_annotator.annotate(frame, detections[mask_ball])

            labels = []
            for i, cls_id in enumerate(detections.class_id):
                if cls_id == self.class_id['player']:
                    labels.append("player")
                elif cls_id == self.class_id['ball']:
                    labels.append("ball")
                else:
                    labels.append("unknown")

            frame = self.label_annotator.annotate(frame, detections, labels=labels)

        return frame

    def track_object(self, frame):
        # Predict separately for ball and player
        ball_detections = self.detect(self.ball_model, frame, 'ball', 0.10)
        player_detections = self.detect(self.player_model, frame, 'player', 0.6)

        # Keep top 22 players
        if player_detections.confidence is not None:
            top_k = np.argsort(-player_detections.confidence)[:22]
            player_detections = player_detections[top_k]

        # Combine detections
        detections = sv.Detections(
            xyxy=np.concatenate([ball_detections.xyxy, player_detections.xyxy]),
            class_id=np.concatenate([ball_detections.class_id, player_detections.class_id]),
            confidence=np.concatenate([ball_detections.confidence, player_detections.confidence])
        )

        annotated = self.draw_detections(frame.copy(), detections)
        cv2.imshow("Detections", annotated)
        cv2.waitKey(1)

        # Return labels for downstream usage
        labels = []
        for xyxy, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            label = 'ball' if cls == self.class_id['ball'] else 'player'
            labels.append({
                'label': label,
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2]
            })

        return labels