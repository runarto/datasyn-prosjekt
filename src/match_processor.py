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

        # Input is a video. We process each frame.

        for idx, path in enumerate(self.image_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[WARN] Could not read image: {path}")
                continue

            annotated = self.process_frame(frame, idx)
            output_path = self.output_dir / path.name
            cv2.imwrite(str(output_path), annotated)


    def play_video (self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
