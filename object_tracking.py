from ultralytics import YOLO
import cv2

class ObjectTracking:
    def __init__(self, image_folder, ball_weights='runs/detect/train_ball/weights/best.pt', player_weights='runs/detect/train_player/weights/best.pt'):
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

