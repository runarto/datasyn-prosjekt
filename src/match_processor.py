import cv2
from pathlib import Path
from src.object_tracking import ObjectTracking, PitchDetection
from src.player_stats import PlayerStats 
import supervision as sv


class MatchProcessor:
    def __init__(self, input_dir: str, output_dir: str, weights_path: str, ssh_mode: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = ObjectTracking(image_folder=str(self.input_dir), weights=weights_path, ssh_mode=ssh_mode)
        self.pitch_detector = PitchDetection()
        self.player_stats = PlayerStats()
        self.ssh_mode = ssh_mode

        self.image_paths = sorted(self.input_dir.glob("*.jpg"))

    def process_frame(self, frame, frame_idx):
        # Pitch detection and homography
        H, _ = self.pitch_detector.get_homography(frame)
        print(H)

        # Run YOLO detection
        results = self.tracker.detect(frame)
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections
        player_detections = detections[detections.class_id == self.tracker.PLAYER_ID]
        player_detections = player_detections.with_nms(threshold=0.5)
        player_detections = self.tracker.tracker.update_with_detections(player_detections)

        ball_detections = detections[detections.class_id == self.tracker.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

        # Annotate image
        annotated_frame = frame.copy()
        self.tracker.ball_annotator.annotate(annotated_frame, ball_detections)
        self.tracker.person_annotator.annotate(annotated_frame, player_detections)
        labels = [f"#{id}" for id in player_detections.tracker_id]
        self.tracker.label_annotator.annotate(annotated_frame, player_detections, labels)

        # Update player stats
        self.player_stats.update(player_detections, H, frame_idx)

        return annotated_frame

    def run(self):
        for idx, path in enumerate(self.image_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[WARN] Could not read image: {path}")
                continue

            annotated = self.process_frame(frame, idx)
            output_path = self.output_dir / path.name
            cv2.imwrite(str(output_path), annotated)

            if not self.ssh_mode:
                cv2.imshow("Annotated Frame", annotated)
                self.player_stats.draw_2d_pitch_map(live=True)
                if cv2.waitKey(30) == ord("q"):
                    break

        if not self.ssh_mode:
            cv2.destroyAllWindows()
            print("[INFO] Final player map:")
            self.player_stats.draw_2d_pitch_map(live=False)