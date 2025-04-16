from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from collections import defaultdict

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
    def __init__(self, image_folder, weights='runs/detect/weights/train/best.pt', ssh_mode=False):
        self.image_folder = image_folder
        self.model = YOLO(weights)
        self.ssh_mode = ssh_mode
        self.fps = 30

        self.BALL_ID = 0
        self.PLAYER_ID = 1

        self.ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),  # gold for ball
            base=20,
            height=17
        )

        self.person_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00FF00']),  # green
            thickness=2
        )

        self.label_annotator = sv.LabelAnnotator(   
            color=sv.Color.from_hex('#FF0000'),  # red
            text_color=sv.Color.from_hex('#000000'),  # black
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.5,
            text_thickness=2
        )

        self.tracker = sv.ByteTrack()


    def detect(self, frame, conf_threshold=0.401):
        results = self.model.predict(frame, conf=conf_threshold)[0]
        return results
    
    def pitch_keypoints(self, frame):
        result = CLIENT.infer(frame, model_id="football-field-detection-f07vi/15") # Pre-trained model for pitch keypoint detection!
        return result

    def track_object(self, frame, frame_name=None):
        self.tracker.reset()
        results = self.detect(frame)
        detections = sv.Detections.from_ultralytics(results)

        ball_detection = detections[detections.class_id == self.BALL_ID]
        ball_detection.xyxy = sv.pad_boxes(xyxy=ball_detection.xyxy, px=10)
        player_detection = detections[detections.class_id == self.PLAYER_ID]
        player_detection = player_detection.with_nms(threshold=0.5)

        player_detection = self.tracker.update_with_detections(detections=player_detection)

        labels = [
            f"#{tracker_id}"
            for tracker_id 
            in player_detection.tracker_id
        ]

        annotated_frame = frame.copy()
        self.ball_annotator.annotate(
            annotated_frame,
            ball_detection,
        )
        self.person_annotator.annotate(
            annotated_frame,
            player_detection,
        )

        self.label_annotator.annotate(
            annotated_frame,
            player_detection,
            labels,
        )


        return annotated_frame


class PitchDetection:
    def __init__(self):
        self.model_id = "football-field-detection-f07vi/15"
        self.prev_homography = None  # Stores last good H

        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF0000'),  # red
            radius=8
        )

        # Reference field points (example: FIFA corners in meters)
        self.world_reference_points = pitch_keypoints_meters.copy()

    def detect_pitch(self, frame):
        result = CLIENT.infer(frame, model_id=self.model_id)
        keypoints = sv.KeyPoints.from_inference(result)
        return keypoints

    def compute_homography(self, keypoints, min_confidence=0.3):
        xy = keypoints.xy[0]  # shape: (32, 2)
        conf = keypoints.confidence[0]  # shape: (32,)

        if xy.shape[0] < 4 or np.sum(conf > min_confidence) < 4:
            return self.prev_homography, True

        # Get top-4 keypoints by confidence
        conf_flat = conf.flatten()  # shape: (32,)
        top_indices = np.argsort(-conf_flat)[:4]  # highest confidence indices

        try:
            image_points = xy[top_indices].astype(np.float32)
            world_points = self.world_reference_points[top_indices].astype(np.float32)
        except IndexError:
            print(f"[ERROR] Tried to access index out of range. Detected {xy.shape[0]} keypoints.")
            return self.prev_homography, True

        # Estimate homography
        H, _ = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)

        if H is not None:
            self.prev_homography = H
            return H, False
        else:
            return self.prev_homography, True


    def get_homography(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        keypoints = self.detect_pitch(rgb_frame)

        # Compute homography
        H, used_fallback = self.compute_homography(keypoints)

        return H, used_fallback
        


