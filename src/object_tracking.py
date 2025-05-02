from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from src.player_stats import PlayerStats
import logging
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker

pitch_keypoints_meters = np.array(
    [
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
        (30.692, 33.500),
    ],
    dtype=np.float32,
)




import numpy as np
from ultralytics import YOLO
import supervision as sv
import cv2
from filterpy.kalman import KalmanFilter
from src.player_stats import PlayerStats  # assumes you have this module

class Track:
    def __init__(self, weights='rbk_detector/aug_balls/weights/best.pt'):
        self.model = YOLO(weights)
        self.fps = 30

        self.BALL_ID = 0
        self.PLAYER_ID = 1

        self.ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'), base=20, height=17
        )

        self.person_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00FF00']), thickness=2
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

        # Kalman Filter for ball tracking
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000
        self.kf.R *= 5
        self.kf.Q *= 0.01
        self.kf_initialized = False

    def detect(self, frame, conf_threshold=0.6):
        return self.model.predict(frame, conf=conf_threshold, imgsz=1280)[0]

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
        player_detections = player_detections.with_nms(threshold=0.75)
        if len(player_detections) > 23:
            top_indices = np.argsort(-player_detections.confidence)[:23]
            player_detections = player_detections[top_indices]
        return self.tracker.update_with_detections(player_detections)

    def update_kalman(self, detection):
        if detection is not None and len(detection.xyxy) > 0:
            x1, y1, x2, y2 = detection.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            z = np.array([cx, cy])
            if not self.kf_initialized:
                self.kf.x = np.array([cx, cy, 0, 0])
                self.kf_initialized = True
            self.kf.predict()
            self.kf.update(z)
        else:
            self.kf.predict()

        # Return prediction as a bbox
        cx, cy = self.kf.x[0], self.kf.x[1]
        size = 10  # for visualization
        return np.array([[cx - size, cy - size, cx + size, cy + size]])

    def track_object(self, frame, frame_idx):
        results = self.detect(frame)
        detections = sv.Detections.from_ultralytics(results)

        ball_det = self.ball_detection(detections)
        tracked_players = self.player_detection(detections)

        # Update Kalman filter for ball
        predicted_ball = self.update_kalman(ball_det)
        tracked_ball = sv.Detections(
            xyxy=predicted_ball,
            confidence=np.array([1.0]),
            class_id=np.array([self.BALL_ID])
        )

        # Annotate
        annotated = frame.copy()
        self.ball_annotator.annotate(annotated, tracked_ball)

        # Annotate players and IDs
        if len(tracked_players.xyxy) > 0:
            self.person_annotator.annotate(annotated, tracked_players)
            labels = [f"#{tid}" for tid in tracked_players.tracker_id]
            self.label_annotator.annotate(annotated, tracked_players, labels)

        return annotated




class PitchDetection:
    def __init__(self, smoothing: float = 0.7):
        """
        PitchDetection handles football pitch keypoint detection and homography estimation.
        :param smoothing: Exponential smoothing factor for homography updates (0 < smoothing <= 1).
        """
        self.model = YOLO(
            '/home/runarto/Documents/datasyn-prosjekt-final/src/pitch_keypoint_model/final_train/weights/best.pt'
        )
        self.prev_homography = None
        self.smoothing = smoothing  # weight for new homography in smoothing

        # Annotator for debugging keypoints
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF0000'), radius=8
        )

        # Real-world coordinates of pitch keypoints (same order as model outputs)
        self.world_reference_points = pitch_keypoints_meters.copy()

    def annotate_keypoints(self, frame: np.ndarray, keypoints: sv.KeyPoints, min_confidence=0.3) -> np.ndarray:
        """Draw high-confidence keypoints on the frame."""
        annotated = frame.copy()
        xy = keypoints.xy[0]
        conf = keypoints.confidence[0]
        for i, (pt, c) in enumerate(zip(xy, conf)):
            if c > min_confidence:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    annotated, str(i), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1
                )
        return annotated

    def detect_pitch(self, frame: np.ndarray) -> sv.KeyPoints:
        """Run keypoint model and return KeyPoints object."""
        result = self.model(frame, verbose=False)
        return sv.KeyPoints.from_ultralytics(result)

    def compute_homography(
        self, keypoints: sv.KeyPoints,
        min_confidence: float = 0.8,
        collinearity_thresh: float = 1.0
    ) -> tuple[np.ndarray, bool]:
        """
        Compute and smooth homography; fallback if insufficient or collinear.
        Returns (H, used_fallback).
        """
        # No keypoints
        if keypoints is None or len(keypoints.xy) == 0:
            logging.warning('[WARN] No keypoints; using previous homography.')
            return self.prev_homography, True
        xy = keypoints.xy[0]
        conf = keypoints.confidence[0].flatten()
        # Filter by confidence
        idx = np.where(conf > min_confidence)[0]
        if len(idx) < 4:
            logging.warning(f'[WARN] Only {len(idx)} strong keypoints; need >=4. Fallback.')
            return self.prev_homography, True
        img_pts = xy[idx].astype(np.float32)
        world_pts = self.world_reference_points[idx].astype(np.float32)
        # Check non-collinearity via convex hull area
        hull = cv2.convexHull(img_pts)
        area = cv2.contourArea(hull)
        if area < collinearity_thresh:
            logging.warning(f'[WARN] Keypoints nearly collinear (area={area:.2f}); fallback.')
            return self.prev_homography, True
        # RANSAC homography
        H_new, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)
        # Validate
        if H_new is None or np.isnan(H_new).any() or abs(np.linalg.det(H_new)) < 1e-6:
            logging.warning('[WARN] Invalid homography; fallback.')
            return self.prev_homography, True
        # Smooth
        if self.prev_homography is None:
            H = H_new
        else:
            H = self.smoothing * H_new + (1 - self.smoothing) * self.prev_homography
        self.prev_homography = H
        return H, False

    def get_homography(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Public: detect keypoints and compute homography."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps = self.detect_pitch(rgb)
        return self.compute_homography(kps)



def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return x_center, y_center

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]