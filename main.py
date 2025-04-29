import os
import socket
from src.match_processor import MatchProcessor
import cv2

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

BASE_DIR = "/datasets/tdt4265/other/rbk"
FINAL_VAL_DIR = os.path.join(BASE_DIR, '4_annotate_1min_bodo_start')
OUTPUT_DIR = "output"
WEIGHTS = "src/rbk_detector/aug_balls/weights/best.pt"

def main():

    processor = MatchProcessor(
        input_dir=FINAL_VAL_DIR,
        output_dir=OUTPUT_DIR,
        weights_path=WEIGHTS,
    )
    processor.run()


if __name__ == "__main__":
    main()

