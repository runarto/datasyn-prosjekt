import os
import yaml
import shutil
from PIL import Image
import random
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from ultralytics import YOLO
import torch
import time
import socket


class Model:
    def __init__(self, base_dir='your_dataset', crop_dir='ball_crops'):
        self.base_dir = base_dir
        self.crop_dir = crop_dir

        hostname = socket.gethostname()

        if "clab" in hostname:
            self.datasets = [
                ('/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1',
                    '/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt'),

                ('/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/img1',
                    '/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/gt/gt.txt')
            ]
        else:
            self.datasets = [
                ('/Users/runartobiassen/Downloads/1_train-val_1min_aalesund_from_start/img1',
                    '/Users/runartobiassen/Downloads/1_train-val_1min_aalesund_from_start/gt/gt.txt'),

                ('/Users/runartobiassen/Downloads/2_train-val_1min_after_goal/img1',
                    '/Users/runartobiassen/Downloads/2_train-val_1min_after_goal/gt/gt.txt')
        ]

    def create_dataset_yaml(self, yaml_path='your_dataset/rbk.yaml'):
        abs_dataset_path = os.path.abspath(self.base_dir)
        config = {
            'path': abs_dataset_path,
            'train': os.path.join(abs_dataset_path, 'images/train'),
            'val': os.path.join(abs_dataset_path, 'images/val'),
            'nc': 2,
            'names': ['ball', 'player']
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f"✅ Created dataset config at {yaml_path}")
        return yaml_path

    def prepare_data_structure(self, force_override=False):
        if os.path.exists(self.base_dir) and not force_override:
            print("⚠️ Directory already exists. Please remove or rename it.")
            return False

        shutil.rmtree(self.base_dir, ignore_errors=True)
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.base_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, 'labels', split), exist_ok=True)
        print("✅ Directory structure ensured.")
        return True

    def split_data_by_class_presence(self, img_src, gt_txt, train_ratio=0.8, dataset_id=1):
        img_files = sorted([f for f in os.listdir(img_src) if f.endswith('.jpg')])
        frame_to_img = {int(Path(f).stem): f for f in img_files}
        img_w, img_h = 1920, 1080

        try:
            with Image.open(os.path.join(img_src, img_files[0])) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"⚠️ Could not read image dimensions. Using fallback 1920x1080. Error: {e}")

        labels = {}
        ball_frames = set()
        with open(gt_txt, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9 or float(parts[8]) == 0.0:
                    continue
                frame = int(parts[0])
                cls = 0 if int(parts[7]) == 1 else 1
                x, y, w, h = map(float, parts[2:6])
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                entry = f"{cls} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}"
                labels.setdefault(frame, []).append(entry)
                if cls == 0:
                    ball_frames.add(frame)

        ball_frames = list(ball_frames)
        random.shuffle(ball_frames)
        split_index = int(train_ratio * len(ball_frames))
        train_frames = set(ball_frames[:split_index])
        val_frames = set(ball_frames[split_index:])

        for frame, entries in labels.items():
            if frame not in frame_to_img:
                continue
            split = 'train' if frame in train_frames else 'val'
            output_img_dir = Path(self.base_dir) / 'images' / split
            output_lbl_dir = Path(self.base_dir) / 'labels' / split

            img_file = frame_to_img[frame]
            new_filename = f"{dataset_id}_{frame:06d}"

            shutil.copy2(os.path.join(img_src, img_file), output_img_dir / f"{new_filename}.jpg")
            with open(output_lbl_dir / f"{new_filename}.txt", 'w') as f:
                f.write('\n'.join(entries) + '\n')

        print(f"✅ Copied {len(train_frames)} train frames and {len(val_frames)} val frames.")


    def extract_ball_crops(self, dataset_id):
        crop_dir = Path(self.crop_dir) / f"dataset_{dataset_id}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        img_dir = Path(self.base_dir) / 'images/train'
        label_dir = Path(self.base_dir) / 'labels/train'

        for label_path in glob(f"{label_dir}/*.txt"):
            frame_id = Path(label_path).stem
            img_path = img_dir / f"{frame_id}.jpg"
            if not img_path.exists():
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            img = Image.open(img_path)
            w, h = img.size

            for i, line in enumerate(lines):
                cls, x, y, bw, bh = map(float, line.strip().split())
                if int(cls) != 0:
                    continue

                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                expand_px = random.randint(5, 15)
                x1 = max(0, x1 - expand_px)
                y1 = max(0, y1 - expand_px)
                x2 = min(w, x2 + expand_px)
                y2 = min(h, y2 + expand_px)

                ball_crop = img.crop((x1, y1, x2, y2))
                ball_crop.save(crop_dir / f"{frame_id}_{i}.png")

        print(f"✅ Saved ball crops for dataset {dataset_id} to '{crop_dir}'")

    def add_synthetic_balls(self, img_path, label_path, dataset_id, num_augmentations=4):
        img = cv2.imread(img_path)
        if img is None:
            return

        h, w = img.shape[:2]
        crop_dir = Path(self.crop_dir) / f"dataset_{dataset_id}"
        crop_paths = list(crop_dir.glob("*.png"))

        if not crop_paths:
            return

        with open(label_path, 'r') as f:
            existing_labels = [line.strip() for line in f.readlines() if line.strip()]

        player_boxes = []
        for line in existing_labels:
            cls, cx, cy, bw, bh = map(float, line.split())
            if int(cls) == 1:
                px1 = int((cx - bw/2) * w)
                py1 = int((cy - bh/2) * h)
                px2 = int((cx + bw/2) * w)
                py2 = int((cy + bh/2) * h)
                player_boxes.append((px1, py1, px2, py2))

        new_labels = existing_labels.copy()
        added_balls = 0

        for _ in range(num_augmentations * 2):
            if added_balls >= num_augmentations:
                break

            crop_path = random.choice(crop_paths)
            crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if crop is None:
                continue

            orig_h, orig_w = crop.shape[:2]
            base_size = (orig_w * orig_h) ** 0.5
            target_base = 40
            base_scale = target_base / base_size
            scale_variation = random.uniform(0.8, 1.2)
            scale = max(1.0, min(3.0, base_scale * scale_variation))

            new_w = min(int(orig_w * scale), int(w * 0.1))
            new_h = min(int(orig_h * scale), int(h * 0.1))

            interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

            for _ in range(20):
                x = random.randint(0, w - new_w)
                y = random.randint(int(h * 0.6), h - new_h)

                overlaps = any(
                    (x < px2 and x + new_w > px1 and y < py2 and y + new_h > py1)
                    for (px1, py1, px2, py2) in player_boxes
                )

                if not overlaps:
                    alpha = random.uniform(0.8, 1.2)
                    beta = random.uniform(-20, 20)
                    crop_varied = cv2.convertScaleAbs(crop_resized, alpha=alpha, beta=beta)
                    mask = np.ones(crop_varied.shape, dtype=np.float32) * 0.7
                    img[y:y+new_h, x:x+new_w] = (
                        img[y:y+new_h, x:x+new_w] * (1 - mask) + crop_varied * mask
                    ).astype(np.uint8)

                    cx = (x + new_w / 2) / w
                    cy = (y + new_h / 2) / h
                    norm_w = new_w / w
                    norm_h = new_h / h
                    new_labels.append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
                    added_balls += 1
                    break

        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            f.write('\n'.join(new_labels) + '\n')

    def process_dataset(self, train_ratio=0.8):
        for idx, (img_src, gt_txt) in enumerate(self.datasets, start=1):
            print(f"\n📦 Processing dataset {idx}: {img_src}")
            self.split_data_by_class_presence(img_src, gt_txt, train_ratio, dataset_id=idx)
            self.extract_ball_crops(dataset_id=idx)

            train_img_dir = Path(self.base_dir) / 'images/train'
            train_label_dir = Path(self.base_dir) / 'labels/train'

            for img_path in train_img_dir.glob("*.jpg"):
                label_path = train_label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    self.add_synthetic_balls(str(img_path), str(label_path), dataset_id=idx)

            print(f"✅ Completed processing for dataset {idx}")

    def estimate_energy_and_distance(self, compute_seconds, gpu_power_watts=450, tesla_efficiency_kwh_per_100km=17):
        hours = compute_seconds / 3600
        energy_kwh = (gpu_power_watts * hours) / 1000  # Watts × hours → kWh
        distance_km = (energy_kwh / tesla_efficiency_kwh_per_100km) * 100
        return energy_kwh, distance_km
    
    def tune_hyperparameters(self, data_yaml, weights=None):
        model = YOLO(weights) if weights else YOLO('yolov8s.pt')
        # Range of hyperparameters to tune

        search_space = {
            'lr0': (1e-5, 1e-1),        # initial learning rate
            'momentum': (0.7, 0.98),
            'weight_decay': (0.0, 0.001),
            'box': (5.0, 20.0),
            'cls': (0.3, 3.0),
            'dfl': (0.5, 3.0),
            'hsv_h': (0.0, 0.1),
            'hsv_s': (0.5, 0.9),
            'hsv_v': (0.3, 0.7),
            'translate': (0.0, 0.2),
            'scale': (0.5, 1.5),
            'fliplr': (0.0, 0.5),
            'mosaic': (0.2, 1.0)
        }

        # Start tuning
        model.tune(
            data='your_dataset/rbk.yaml',  # your dataset YAML
            epochs=30,                     # small epochs per candidate
            iterations=300,                # number of hyperparameter sets to try
            space=search_space,            # search space
            optimizer='AdamW',             # or SGD, Adam, AdamW
            plots=True,
            save=True,
            val=True
        )


    def train_pitch_keypoint_model(self, data_yaml, weights=None):
        model = YOLO(weights) if weights else YOLO('yolov8x-pose.pt')  # Use YOLOv8x pre-trained pose model

        results = model.train(
            task='pose',                    # Set the task to 'pose' for keypoint detection
            data=data_yaml,
            epochs=100,                     # Train for 100 epochs
            batch=4,                        # Smaller batch size due to memory constraints
            imgsz=640,                      # Image size of 640x640
            mosaic=0.0,                     # Mosaic augmentation is disabled (not useful for keypoints)
            plots=True,                     # Save plots of training metrics
            device='0' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
            project='pitch_keypoint_model', # Save outputs in a dedicated project folder
            name='final_train',              # Training run name
            verbose=True,
        )
        print("✅ Training complete. Model saved in:", results.save_dir)


    def train_model(self, data_yaml, weights=None, use_pretrained=False):
        model = YOLO(weights) if use_pretrained and weights else YOLO('yolov8s.pt')

        with open('best_hyp.yaml', 'r') as f:
            best_hyp = yaml.safe_load(f)

        results = model.train(
            data=data_yaml,
            epochs=100,
            batch=8,
            imgsz=1280,
            plots=True,
            save_period=5,
            device='0' if torch.cuda.is_available() else 'cpu',
            project='rbk_detector',
            name='train8s',
            verbose=True,
            patience=15,
            cos_lr=True,
            cache=True,
            **best_hyp,   # 🚀 MAGIC - unpack best hyperparameters into the training!
        )


        print("✅ Training complete. Model saved in:", results.save_dir)


if __name__ == "__main__":
    model = Model(base_dir='../your_dataset', crop_dir='../ball_crops')
    flag = False
    train_pitch = False
    train_ball_player = True

    dataset_yaml = model.create_dataset_yaml()
    if flag:
        model.prepare_data_structure(force_override=True)
        model.process_dataset(train_ratio=0.8)
        
    if os.path.exists('../ball_crops'):
        shutil.rmtree('../ball_crops') # This is just to not exceed the quota on clab

    start_time = time.time()
    if train_pitch:
        dataset_yaml = "../football-field-detection/data.yaml"
        model.train_pitch_keypoint_model(data_yaml=dataset_yaml, weights=None)

    
    if train_ball_player:
        model.train_model(data_yaml=dataset_yaml, weights=None, use_pretrained=False)

    compute_seconds = time.time() - start_time
    energy_kwh, distance_km = model.estimate_energy_and_distance(compute_seconds)


    print(f"🕒 Total compute time: {compute_seconds/60:.2f} minutes")
    print(f"⚡ Estimated energy used: {energy_kwh:.2f} kWh")
    print(f"🚗 Equivalent Tesla Model Y driving distance: {distance_km:.2f} km")