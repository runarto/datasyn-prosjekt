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
        self.fps = 30

        self.tracker = ObjectTracking(image_folder=str(self.input_dir), weights=weights_path, ssh_mode=ssh_mode)
        self.pitch_detector = PitchDetection()
        self.player_stats = PlayerStats()
        self.ssh_mode = ssh_mode

        self.image_paths = sorted(self.input_dir.glob("*.jpg"))

    def process_frame(self, frame, frame_idx):
        # Pitch detection and homography
        H, _ = self.pitch_detector.get_homography(frame)
        annotated_frame = self.tracker.track_object(frame, frame_idx, H)
        return annotated_frame

    def run(self):
        cap = cv2.VideoCapture(self.input_dir)

        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.input_dir}")
            return

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1', etc.

        # Output video path
        output_video_path = self.output_dir / f"{Path(self.input_dir).stem}_annotated.mp4"
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.process_frame(frame, frame_idx)
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"[INFO] Finished processing. Saved annotated video to {output_video_path}")




