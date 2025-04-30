from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from src.player_stats import PlayerStats
import logging

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

class ObjectTracking:
    def __init__(
        self,
        image_folder,
        weights="rbk_detector/aug_balls/weights/best.pt",
        ssh_mode=False,
    ):
        self.image_folder = image_folder
        self.model = YOLO(weights)        # your detection model
        self.ssh_mode = ssh_mode
        self.fps = 30

        # class IDs in your model
        self.BALL_ID = 0
        self.PLAYER_ID = 1

        # drawing helpers
        self.ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=20, height=17
        )
        self.person_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#00FF00"]), thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.from_hex("#FF0000"),
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.35,
            text_thickness=1,
            text_padding=2,
        )

        # stats tracker for speed, etc.
        self.stats = PlayerStats(fps=self.fps)

    def track_object(self, frame, frame_idx, homography):
        """
        Detects & tracks players + ball in a single frame, then annotates.
        Uses Ultralytics’ built-in BoT-SORT tracker for ID consistency.
        """
        # Run detection + BoT-SORT in one call
        results = self.model.track(
            source=[frame],           # single image
            tracker="botsort.yaml",   # use the official BoT-SORT config
            conf=0.6,                 # same minimum confidence
            show=False                # we’ll handle drawing ourselves
        )[0]

        # Extract boxes, track IDs, and classes
        boxes   = results.boxes.xyxy.cpu().numpy()        # shape: [N,4]
        ids     = results.boxes.id.cpu().numpy().astype(int)   # shape: [N]
        classes = results.boxes.cls.cpu().numpy().astype(int)  # shape: [N]

        annotated = frame.copy()

        for (x1, y1, x2, y2), track_id, cls in zip(boxes, ids, classes):
            bbox = np.array([int(x1), int(y1), int(x2), int(y2)])

            if cls == self.PLAYER_ID:
                # draw player ellipse
                self.person_annotator.annotate(
                    annotated, sv.Detections(xyxy=[bbox])
                )

                # compute & draw speed label
                _, speed, _ = self.stats.compute_player_stats(track_id)
                label = f"#{track_id} | {speed:.2f}"
                self.label_annotator.annotate(
                    annotated,
                    sv.Detections(xyxy=[bbox]),
                    [label]
                )

            elif cls == self.BALL_ID:
                # draw ball triangle
                self.ball_annotator.annotate(
                    annotated, sv.Detections(xyxy=[bbox])
                )

        return annotated


class PitchDetection:
    def __init__(self):
        self.model = YOLO("src/pitch_keypoint_model/final_train/weights/best.pt")
        self.prev_homography = None

        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex("#FF0000"), radius=8
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
                cv2.putText(
                    annotated,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return annotated

    def detect_pitch(self, frame):
        result = self.model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        return keypoints

    def compute_homography(self, keypoints, min_confidence=0.8):
        if keypoints is None or len(keypoints.xy) == 0:
            logging.warning("[WARN] No keypoints detected; using previous homography.")
            return self.prev_homography, True
        xy = keypoints.xy[0]  # shape: [N, 2]
        conf = keypoints.confidence[0].flatten()  # shape: [N,]

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
        H, _ = cv2.findHomography(image_points, world_points, cv2.RANSAC)

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
