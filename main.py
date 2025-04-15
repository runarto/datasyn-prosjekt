import os
import cv2
from object_tracking import ObjectTracking  # Make sure this points to the updated class
from util import copy_random_images

if __name__ == "__main__":
    image_folder = "/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1"

    copy_random_images(image_folder, "example_images", num_images=20)

    # tracker = ObjectTracking(
    #     image_folder=image_folder,
    #     ball_weights="ball_train/weights/best.pt",
    #     player_weights="player_train/weights/best.pt",
    #     auto_calibrate=True
    # )

    # image_files = sorted([
    #     f for f in os.listdir(image_folder)
    #     if f.endswith(('.jpg', '.png'))
    # ])

    # print(f"Processing {len(image_files)} frames... (press ENTER to step through, Q to quit)")

    # for idx, image_file in enumerate(image_files):
    #     image_path = os.path.join(image_folder, image_file)
    #     frame = cv2.imread(image_path)

    #     if frame is None:
    #         print(f"[Warning] Failed to load {image_file}")
    #         continue

    #     labels = tracker.track_object(frame)

    #     key = cv2.waitKey(0) & 0xFF  # Wait for user input
    #     if key == ord('q'):
    #         print("Exiting early.")
    #         break

    # cv2.destroyAllWindows()
    # print("Done.")
