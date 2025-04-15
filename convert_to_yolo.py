import os
import shutil
import random

# === CONFIG ===
data_sources = [
    {
        "gt_file": "/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt",
        "image_folder": "/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1"
    },
    {
        "gt_file": "/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/gt/gt.txt",
        "image_folder": "/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/img1"
    }
]

output_dir = "your_dataset"
image_width = 1920
image_height = 1080
train_ratio = 0.8

# === Prepare folders ===
img_out_train = os.path.join(output_dir, "images", "train")
img_out_val = os.path.join(output_dir, "images", "val")
label_out_train = os.path.join(output_dir, "labels", "train")
label_out_val = os.path.join(output_dir, "labels", "val")

for folder in [img_out_train, img_out_val, label_out_train, label_out_val]:
    os.makedirs(folder, exist_ok=True)

annotations = {}

# === Parse all GT.txt files ===
for idx, source in enumerate(data_sources):
    with open(source["gt_file"], "r") as f:
        lines = f.readlines()

    for line in lines:
        items = line.strip().split(",")
        frame = int(items[0])
        x, y, w, h = map(float, items[2:6])
        class_id = int(items[7])  # 1 = ball, 2 = player
        visibility = float(items[8])

        if visibility == 0:
            continue

        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        w_norm = w / image_width
        h_norm = h / image_height

        yolo_class = class_id - 1  # 0 = ball, 1 = player

        prefix = str(idx + 1)  # use dataset index as prefix
        filename = f"{prefix}_{frame:06d}"
        if filename not in annotations:
            annotations[filename] = {
                "labels": [],
                "image_path": os.path.join(source["image_folder"], f"{frame:06d}.jpg")
            }
        annotations[filename]["labels"].append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

# === Split into train/val and write files ===
all_filenames = sorted(list(annotations.keys()))
random.seed(42)
random.shuffle(all_filenames)
split_idx = int(len(all_filenames) * train_ratio)

train_set = set(all_filenames[:split_idx])
val_set = set(all_filenames[split_idx:])

for fname, data in annotations.items():
    src_img = data["image_path"]

    if not os.path.exists(src_img):
        print(f"⚠️ Image not found: {src_img}")
        continue

    if fname in train_set:
        label_path = os.path.join(label_out_train, f"{fname}.txt")
        img_dest = os.path.join(img_out_train, f"{fname}.jpg")
    else:
        label_path = os.path.join(label_out_val, f"{fname}.txt")
        img_dest = os.path.join(img_out_val, f"{fname}.jpg")

    shutil.copy2(src_img, img_dest)

    with open(label_path, "w") as f:
        f.write("\n".join(data["labels"]) + "\n")

print(f"✅ Wrote {len(train_set)} training and {len(val_set)} validation samples to '{output_dir}'")
