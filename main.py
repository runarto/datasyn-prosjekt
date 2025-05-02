import os
from src.match_processor import MatchProcessor
import cv2
import shutil
from src.train_model import Train


def frames_to_video():
    image_folder = 'output'
    video_name = 'output.mp4'  # Match codec with extension
    fps = 30  # Frames per second

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        raise ValueError(f"No .jpg images found in the folder: {image_folder}")

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        raise ValueError(f"Failed to read the first image: {images[0]}")

    height, width, _ = frame.shape
    print(f"Frame dimensions: {width}x{height}")

    # Important: Use correct codec for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    if not video.isOpened():
        raise IOError("VideoWriter failed to open. Check codec and output path.")

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        if frame is None:
            print(f"Warning: Skipping invalid image {image}")
            continue
        # Resize if needed (shouldn't happen ideally)
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print(f"Video saved as {video_name}")


# --- USER ---
USER = os.getlogin()

# --- Paths ---
BASE_DIR = f"/work/{USER}/data/video"
OUTPUT_DIR = "output"

# Source video (on shared dataset storage)
SOURCE_VIDEO = "/datasets/tdt4265/other/rbk/4_annotate_1min_bodo_start/4_annotate_1min_bodo_start.mp4"

# Target video (local working directory)
VIDEO_FILENAME = os.path.basename(SOURCE_VIDEO)
VIDEO_TO_ANNOTATE = os.path.join(BASE_DIR, VIDEO_FILENAME)

# --- Ensure video is copied ---
os.makedirs(BASE_DIR, exist_ok=True)
if not os.path.exists(VIDEO_TO_ANNOTATE):
    print(f"[INFO] Copying video from {SOURCE_VIDEO} to {VIDEO_TO_ANNOTATE}...")
    shutil.copy2(SOURCE_VIDEO, VIDEO_TO_ANNOTATE)
else:
    print(f"[INFO] Video already exists at {VIDEO_TO_ANNOTATE}")


def main():

    # Make sure the paths are correct
    train = False
    if train:
        trainer = Train()
        trainer.run()

    WEIGHTS = f"/home/{USER}/Documents/datasyn-prosjekt-final/src/ball_player_model/phase2_full/weights/best.pt"

    processor = MatchProcessor(
        video=VIDEO_TO_ANNOTATE,
        output_dir=OUTPUT_DIR,
        weights_path=WEIGHTS,
    )
    processor.run()


if __name__ == "__main__":
    main()

