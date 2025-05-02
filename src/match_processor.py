import cv2
from pathlib import Path
from src.object_tracking import Track, PitchDetection
from src.player_stats import PlayerStats 
import supervision as sv


class MatchProcessor:
    def __init__(self, video: str, output_dir: str, weights_path: str):

        # Video path
        self.video = Path(video)
     
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracker model
        self.tracker = Track(weights=weights_path)

    def process_frame(self, frame, frame_idx):
        """Process a single frame for object tracking, and annotates it."""
        annotated_frame = self.tracker.track_object(frame, frame_idx)
        return annotated_frame

    def run(self):
        """Main method to run the video processing. Loops through the video frames, processes them, and saves the annotated video."""
        cap = cv2.VideoCapture(str(self.video))

        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {self.video}")
            return

        # Video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        # Output video path
        output_video_path = self.output_dir / f"{Path(self.video).stem}_annotated.mp4"
        out = cv2.VideoWriter(str(output_video_path), fourcc, self.fps, (width, height))

        frame_idx = 0

        # Main loop to read frames from the video
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




