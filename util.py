import os
import cv2
import shutil
import random

def convert_and_split(gt_path, image_folder, shared_image_dir, label_dir_ball, label_dir_player, train_ratio=0.8):
    temp_label_ball_dir = os.path.join(label_dir_ball, "all_labels")
    temp_label_player_dir = os.path.join(label_dir_player, "all_labels")
    os.makedirs(temp_label_ball_dir, exist_ok=True)
    os.makedirs(temp_label_player_dir, exist_ok=True)

    with open(gt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        class_id = int(parts[7])
        visibility = float(parts[8])

        if visibility == 0.0:
            continue

        x, y, w, h = map(float, parts[2:6])
        image_name = f"{frame_id:06}.jpg"
        image_path = os.path.join(image_folder, image_name)

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        label_line = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"

        if class_id == 1:  # Ball
            label_file = os.path.join(temp_label_ball_dir, f"{image_name.replace('.jpg', '.txt')}")
            with open(label_file, 'a') as lf:
                lf.write(label_line)
        elif class_id == 2:  # Player
            label_file = os.path.join(temp_label_player_dir, f"{image_name.replace('.jpg', '.txt')}")
            with open(label_file, 'a') as lf:
                lf.write(label_line)

        # Copy image to shared image pool if not already copied
        for split in ['train', 'val']:
            shared_train_path = os.path.join(shared_image_dir, split, image_name)
            if not os.path.exists(shared_train_path):
                os.makedirs(os.path.join(shared_image_dir, split), exist_ok=True)
                shutil.copyfile(image_path, shared_train_path)
                break  # copy once only

    # Split image files once per full dataset
    all_images = sorted([f for f in os.listdir(shared_image_dir + '/train') if f.endswith(('.jpg', '.png'))] +
                        [f for f in os.listdir(shared_image_dir + '/val') if f.endswith(('.jpg', '.png'))])
    random.seed(42)
    random.shuffle(all_images)
    split_index = int(len(all_images) * train_ratio)
    splits = {'train': all_images[:split_index], 'val': all_images[split_index:]}

    for label_dir, temp_dir in [(label_dir_ball, temp_label_ball_dir), (label_dir_player, temp_label_player_dir)]:
        for split_name, split_list in splits.items():
            label_split_dir = os.path.join(label_dir, 'labels', split_name)
            os.makedirs(label_split_dir, exist_ok=True)

            for image_file in split_list:
                label_file = os.path.splitext(image_file)[0] + '.txt'
                src_lbl = os.path.join(temp_dir, label_file)
                dst_lbl = os.path.join(label_split_dir, label_file)
                if os.path.exists(src_lbl):
                    shutil.copyfile(src_lbl, dst_lbl)
                else:
                    open(dst_lbl, 'w').close()

    return "Merged and split dataset successfully."

# Example call (for multiple sequences, run this multiple times):
convert_and_split(
    gt_path="/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt",
    image_folder="/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1",
    shared_image_dir="/home/runarto/Documents/datasyn-project/shared_images",
    label_dir_ball="/home/runarto/Documents/datasyn-project/dataset_ball",
    label_dir_player="/home/runarto/Documents/datasyn-project/dataset_player"
)

convert_and_split(
    gt_path="/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/gt/gt.txt",
    image_folder="/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/img1",
    shared_image_dir="/home/runarto/Documents/datasyn-project/shared_images",
    label_dir_ball="/home/runarto/Documents/datasyn-project/dataset_ball",
    label_dir_player="/home/runarto/Documents/datasyn-project/dataset_player"
) 





