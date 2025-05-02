from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from src.player_stats import PlayerStats
import logging
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker

PITCH_KEYPOINTS = np.array([
    (0.000, 0.000),
    (0.000, 14.093),
    (0.000, 25.078),
    (0.000, 42.843),
    (0.000, 52.898),
    (0.000, 68.000),
    (4.812, 25.078),
    (4.812, 42.877),
    (9.612, 34.014),
    (17.629, 14.093),
    (17.629, 25.078),
    (17.629, 42.843),
    (17.629, 52.898),
    (52.438, 0.000),
    (52.438, 25.098),
    (52.438, 42.858),
    (52.438, 68.000),
    (87.247, 14.093),
    (87.247, 25.078),
    (87.247, 52.898),
    (95.264, 34.014),
    (99.958, 25.078),
    (99.958, 42.843),
    (105.000, 0.000),
    (105.000, 14.093),
    (105.000, 25.078),
    (105.000, 42.843),
    (105.000, 52.898),
    (105.000, 68.000),
    (46.924, 34.014),
    (63.655, 34.014),
    (30.369, 34.014),
], dtype=np.float32)



IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


class Track:
    def __init__(self, weights='rbk_detector/aug_balls/weights/best.pt'):
        self.model = YOLO(weights)
        self.ball_only_model = YOLO("src/ball_player_model/phase1_ball/weights/best.pt")
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
            color=sv.Color.from_hex('#FF0000'),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.35,
            text_thickness=1,
            text_padding=2
        )

        self.tracker = sv.ByteTrack(lost_track_buffer=50)
        self.stats = PlayerStats(fps=self.fps)
        self.pitch_detector = PitchDetection()
        self.homography = None
        self.prev_homography = None
        self.smoothing = 0.7
        self.pitch_length = 106.5
        self.pitch_width = 67

        # Ball tracking
        self.ball_positions = []
        self.interpolated_positions = []

    def detect(self, frame, conf_threshold=0.6):
        results = self.model.predict(frame, conf=conf_threshold, imgsz=1280)[0]
        return results

    def ball_detection(self, frame, detections):
        ball_detections = detections[detections.class_id == self.BALL_ID]
        if hasattr(ball_detections, "confidence") and len(ball_detections) > 0:
            top_idx = np.argmax(ball_detections.confidence)
            top_conf = ball_detections.confidence[top_idx]
            if top_conf > 0.6:
                return ball_detections[top_idx:top_idx+1]
        # If no ball is detected, run ball-only model

        ball_only_results = self.ball_only_model.predict(frame, conf=0.5, imgsz=1920, iou=0.5, classes=[self.BALL_ID])[0]
        ball_only_detections = sv.Detections.from_ultralytics(ball_only_results)
        if len(ball_only_detections) > 0:
            ball_only_detections = ball_only_detections.with_nms(threshold=0.5)
            if len(ball_only_detections) > 0:
                top_idx = np.argmax(ball_only_detections.confidence)
                top_conf = ball_only_detections.confidence[top_idx]
                if top_conf > 0.6:
                    return ball_only_detections[top_idx:top_idx+1]
        # If no ball is detected in either model, return empty detections
        return sv.Detections.empty()

    def player_detection(self, detections):
        conf_threshold = 0.70
        player_detections = detections[(detections.class_id == self.PLAYER_ID) & (detections.confidence >= conf_threshold)]
        player_detections = player_detections.with_nms(threshold=0.70)
        if len(player_detections) > 23:
            top_indices = np.argsort(-player_detections.confidence)[:25]
            player_detections = player_detections[top_indices]
        tracked_players = self.tracker.update_with_detections(player_detections)
        return tracked_players

    def interpolate_ball(self):
        # Use pandas for interpolation
        import pandas as pd
        df = pd.DataFrame(self.ball_positions, columns=["cx", "cy"])
        df_interp = df.interpolate().bfill()
        self.interpolated_positions = df_interp.values.tolist()

    def track_object(self, frame, frame_idx):

        results = self.detect(frame)
        detections = sv.Detections.from_ultralytics(results)
        ball_detection = self.ball_detection(frame, detections)
        player_detection = self.player_detection(detections)

        # Track ball center or None if not found
        if len(ball_detection) > 0:
            x1, y1, x2, y2 = ball_detection.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self.ball_positions.append([cx, cy])
        else:
            self.ball_positions.append([None, None])

        self.interpolate_ball()  # Interpolate after every update

        annotated_frame = frame.copy()
        self.ball_annotator.annotate(annotated_frame, ball_detection)

        # Annotate interpolated ball (only if original missing)
        if len(ball_detection) == 0 and frame_idx < len(self.interpolated_positions):
            cx, cy = self.interpolated_positions[frame_idx]
            if cx is not None and cy is not None:
                det = sv.Detections(
                    xyxy=np.array([[cx-5, cy-5, cx+5, cy+5]]),
                    class_id=np.array([self.BALL_ID])  # Ensure class_id is a 1D NumPy array
                )
                self.ball_annotator.annotate(annotated_frame, det)

        self.person_annotator.annotate(annotated_frame, player_detection)
        if len(player_detection) > 0 and player_detection.tracker_id is not None:
            labels = [f"#{tid}" for tid in player_detection.tracker_id]
            self.label_annotator.annotate(annotated_frame, player_detection, labels)

        return annotated_frame



class PitchDetection:
    def __init__(self, smoothing: float = 0.7):
        """
        PitchDetection handles football pitch keypoint detection and homography estimation.
        """
        self.model = YOLO(
            '/home/runarto/Documents/datasyn-prosjekt-final/src/pitch_keypoint_model/final_train/weights/best.pt'
        )

        # Annotator for debugging keypoints
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF0000'), radius=8
        )
        # Previous homography for smoothing
        self.H = None

    def annotate_keypoints(self, frame, keypoints) -> np.ndarray:
        """Draw high-confidence keypoints on the frame."""
        min_confidence = 0.8
        annotated = frame.copy()
        xy = keypoints.xy[0]
        conf = keypoints.conf[0]

        for i, (pt, c) in enumerate(zip(xy, conf)):
            if c > min_confidence:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )
        return annotated


    def get_keypoints(self, frame):
        """
        Detects keypoints on the football pitch in the given frame.
        :param frame: Input image frame.
        :return: Keypoints.
        """
        # Convert color scheme to RGB
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        results = self.model.predict(frame, conf=0.5, imgsz=1280)[0]
        keypoints = results.keypoints[0].cpu().numpy()
        return keypoints
    
    def get_homography(self, keypoints):
        """
        Computes the homography matrix from detected keypoints to real-world coordinates.
        :param keypoints: Detected keypoints.
        :return: Homography matrix.
        """

        min_confidence = 0.8
        collinearity_thresh = 1.0

        if keypoints is None or len(keypoints.xy) == 0:
                logging.warning("[WARN] No keypoints detected; using previous homography.")
                logging.warning('[WARN] No keypoints; using previous homography.')
                return None
        
        xy = keypoints.xy[0]  # shape: [N, 2]
        conf = keypoints.conf[0].flatten()  # shape: [N,]
   
        # Filter by confidence
        idx = np.where(conf > min_confidence)[0]
        if len(idx) < 4:
            logging.warning(f'[WARN] Only {len(idx)} strong keypoints; need >=4. Fallback.')
            return NotImplementedError


        img_pts = xy[idx].astype(np.float32)
        world_pts = PITCH_KEYPOINTS[idx].astype(np.float32)
        # Check non-collinearity via convex hull area
        hull = cv2.convexHull(img_pts)
        area = cv2.contourArea(hull)

        if area < collinearity_thresh:
            logging.warning(f'[WARN] Keypoints nearly collinear (area={area:.2f}); fallback.')
            return None
        
        # RANSAC homography
        H_new, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)

        # Validate
        if H_new is None or np.isnan(H_new).any() or abs(np.linalg.det(H_new)) < 1e-6:
            logging.warning('[WARN] Invalid homography; fallback.')
            return None
        
        return H_new



    def detect_and_annotate(self, frame):
        """
        Detects keypoints and computes homography.
        :param frame: Input image frame.
        :return: Homography matrix.
        """
        keypoints = self.get_keypoints(frame)
        H = self.get_homography(keypoints)
        if H is None:
            return None, frame
        self.H = H

        annotated_frame = self.annotate_keypoints(frame, keypoints)

        return H, annotated_frame



