import os
import socket
from src.match_processor import MatchProcessor
import cv2

def frames_to_video():
    image_folder = 'output'
    video_name = 'output.avi'  # Match codec with extension
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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


def main():
    hostname = socket.gethostname()
    play_video = True

    if "clab" in hostname:
        input_file_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/img1"
        ssh_mode = True
    else:
        input_file_path = os.path.expanduser("~/Downloads/3_test_1min_hamkam_from_start/img1")
        ssh_mode = False

    if play_video:
        frames_to_video()
        exit(0)

    output_file_path = "output"
    weights_path = "src/rbk_detector/aug_balls/weights/best.pt"

    processor = MatchProcessor(
        input_dir=input_file_path,
        output_dir=output_file_path,
        weights_path=weights_path,
        ssh_mode=ssh_mode,
    )
    processor.run()


if __name__ == "__main__":
    main()

