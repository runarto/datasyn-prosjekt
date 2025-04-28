from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
from src.player_stats import PlayerStats  # <-- Import PlayerStats
import logging
from pathlib import Path

pitch_keypoints_meters = np.array([
    (0.000, 0.000),
    (0.000, 13.895),
    (0.000, 24.728),
    (0.000, 42.212),
    (0.000, 51.989),
    (0.000, 67.000),
    (4.883, 24.728),
    (4.883, 42.244),
    (9.763, 33.500),
    (17.889, 13.895),
    (17.889, 24.728),
    (17.889, 42.212),
    (17.889, 51.989),
    (53.250, 0.000),
    (53.250, 24.747),
    (53.250, 42.227),
    (53.250, 67.000),
    (88.611, 13.895),
    (88.611, 24.728),
    (88.611, 51.989),
    (96.738, 33.500),
    (101.438, 24.728),
    (101.438, 42.212),
    (106.500, 0.000),
    (106.500, 13.895),
    (106.500, 24.728),
    (106.500, 42.212),
    (106.500, 51.989),
    (106.500, 67.000),
    (47.431, 33.500),
    (64.348, 33.500),
    (30.692, 33.500)
], dtype=np.float32)

load_dotenv()
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

class ObjectTracking:
    def __init__(self, image_folder, weights='rbk_detector/aug_balls/weights/best.pt', ssh_mode=False):
        self.image_folder = image_folder
        self.model = YOLO(weights)
        self.ssh_mode = ssh_mode
        self.fps = 30

        self.BALL_ID = 0
        self.PLAYER_ID = 1

        self.ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=20,
            height=17
        )

        self.person_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00FF00']),
            thickness=2
        )

        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.from_hex('#FF0000'),         # Red bounding box
            text_color=sv.Color.from_hex('#000000'),     # Black text
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.35,                             # Smaller text scale
            text_thickness=1,                            # Thinner text
            text_padding=2                              # Add a little padding around text
        )


        self.tracker = sv.ByteTrack()
        self.stats = PlayerStats(fps=self.fps)

    def detect(self, frame, conf_threshold=0.6):
        results = self.model.predict(frame, conf=conf_threshold)[0]
        return results
    
    def ball_detection(self, detections):
        ball_detections = detections[detections.class_id == self.BALL_ID]
        if hasattr(ball_detections, "confidence") and len(ball_detections) > 0:
            top_idx = np.argmax(ball_detections.confidence)
            top_conf = ball_detections.confidence[top_idx]
            if top_conf > 0.6:
                return ball_detections[top_idx:top_idx+1]
        return sv.Detections.empty()
    
    def player_detection(self, detections):
        conf_threshold = 0.75
        player_detections = detections[(detections.class_id == self.PLAYER_ID) & (detections.confidence >= conf_threshold)]

        # Apply Non-Maximum Suppression (NMS)
        player_detections = player_detections.with_nms(threshold=0.75)

        # Keep only top 23 players if needed
        if len(player_detections) > 23:
            top_indices = np.argsort(-player_detections.confidence)[:23]
            player_detections = player_detections[top_indices]

        # Update tracker and get the tracked detections
        tracked_players = self.tracker.update_with_detections(player_detections)

        return tracked_players

    def track_object(self, frame, frame_idx, homography):
        results = self.detect(frame)
        detections = sv.Detections.from_ultralytics(results)

        ball_detection = self.ball_detection(detections)
        player_detection = self.player_detection(detections)
        self.save_predictions(frame_idx, player_detection, ball_detection, save_dir="predictions")


        # Update player stats (speed)
        self.stats.update(player_detection, homography, frame_idx)

        labels = []
        for tracker_id in player_detection.tracker_id:
            _, speed, _ = self.stats.compute_player_stats(tracker_id)
            labels.append(f"#{tracker_id} | {speed:.2f}")

        annotated_frame = frame.copy()
        self.ball_annotator.annotate(annotated_frame, ball_detection)
        self.person_annotator.annotate(annotated_frame, player_detection)
        self.label_annotator.annotate(annotated_frame, player_detection, labels)

        return annotated_frame
    
    def save_predictions(self, frame_idx, player_detections, ball_detection, save_dir):
        """
        Save player and ball detections to a file for later evaluation.

        Args:
            frame_idx (int): The index of the frame.
            player_detections (sv.Detections): Player detections with tracking IDs.
            ball_detection (sv.Detections): Ball detections.
            save_dir (Path): Directory where to save the prediction files.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        lines = []

        # Save players
        for xyxy, track_id, conf in zip(player_detections.xyxy, player_detections.tracker_id, player_detections.confidence):
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            line = f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},2,{conf:.4f},1"
            lines.append(line)

        # Save ball
        for xyxy, conf in zip(ball_detection.xyxy, ball_detection.confidence):
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            line = f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{conf:.4f},1"
            lines.append(line)

        # Save to file
        pred_file = save_dir / "predictions.txt"
        with open(pred_file, 'a') as f:
            f.write('\n'.join(lines) + '\n')


class PitchDetection:
    def __init__(self):
        self.model = YOLO("src/pitch_keypoint_model/final_train/weights/best.pt")
        self.model_id = "football-field-detection-f07vi/15"
        self.prev_homography = None

        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF0000'),
            radius=8
        )

        self.world_reference_points = pitch_keypoints_meters.copy()

    def annotate_keypoints(self, frame, keypoints, min_confidence=0.3):
        annotated = frame.copy()
        xy = keypoints.xy[0]
        conf = keypoints.confidence[0]

        for i, (pt, c) in enumerate(zip(xy, conf)):
            if c > min_confidence:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(annotated, str(i), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        return annotated

    def detect_pitch(self, frame):
        result = self.model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        return keypoints

    def compute_homography(self, keypoints, min_confidence=0.8):
        if keypoints is None or len(keypoints.xy) == 0:
            logging.warning("[WARN] No keypoints detected; using previous homography.")
            return self.prev_homography, True
        xy   = keypoints.xy[0]            # shape: [N, 2]
        conf = keypoints.confidence[0].  flatten()  # shape: [N,]

        # Find all indices whose confidence exceeds threshold
        valid_indices = np.where(conf > min_confidence)[0]

        # Need at least 4 points to solve homography
        if len(valid_indices) < 4:
            # not enough points → fallback
            return self.prev_homography, True

        try:
            image_points = xy[valid_indices].astype(np.float32)
            world_points = self.world_reference_points[valid_indices].astype(np.float32)
        except IndexError:
            logging.error(
                f"[ERROR] Keypoint/world‐point size mismatch:"
                f" got {xy.shape[0]} keypoints but {len(self.world_reference_points)} world refs"
            )
            return self.prev_homography, True

        # RANSAC homography using *all* high-confidence correspondences
        H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC)

        # Validate result
        if H is not None and not np.isnan(H).any() and abs(np.linalg.det(H)) > 1e-6:
            self.prev_homography = H
            return H, False
        else:
            logging.warning("[WARN] Homography unstable/invalid; using previous.")
            return self.prev_homography, True



    def get_homography(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = self.detect_pitch(rgb_frame)
        H, used_fallback = self.compute_homography(keypoints)
        return H, used_fallback