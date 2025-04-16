import os
import yaml
import shutil
from PIL import Image
from ultralytics import YOLO
import torch


def create_dataset_yaml(yaml_path='your_dataset/rbk.yaml'):
    config = {
        'path': os.path.abspath('your_dataset'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['ball', 'player']
    }
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Created dataset config at {yaml_path}")
    return yaml_path


def prepare_data_structure():
    base_dir = 'your_dataset'
    if os.path.exists(base_dir):
        print("⚠️ Directory already exists. Please remove or rename it.")
        return False
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)
    print("✅ Directory structure ensured.")
    return True


def copy_images_and_labels(img_src, gt_txt, split='train'):
    from pathlib import Path
    output_img_dir = Path(f'your_dataset/images/{split}')
    output_lbl_dir = Path(f'your_dataset/labels/{split}')
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Read annotations
    img_files = sorted([f for f in os.listdir(img_src) if f.endswith('.jpg')])
    frame_to_img = {int(Path(f).stem): f for f in img_files}

    img_w, img_h = 1920, 1080
    try:
        with Image.open(os.path.join(img_src, img_files[0])) as im:
            img_w, img_h = im.size
    except:
        print("⚠️ Could not read image dimensions. Falling back to 1920x1080.")

    labels = {}
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

    for frame, entries in labels.items():
        if frame in frame_to_img:
            img_file = frame_to_img[frame]
            shutil.copy2(os.path.join(img_src, img_file), output_img_dir / img_file)
            with open(output_lbl_dir / f"{frame:06d}.txt", 'w') as f:
                f.write('\n'.join(entries) + '\n')

    print(f"✅ Copied {len(labels)} images and labels to {split} set.")


def train_model(data_yaml, weights='runs/soccer/rbk_detector/weights/best.pt', use_pretrained=True):
    if use_pretrained:
        model = YOLO(weights)
    else:
        model = YOLO('yolov8s.pt')
    results = model.train(
        data=data_yaml,
        epochs=50,
        batch=4,
        imgsz=1280,
        plots=True,
        save_period=10,
        device='0' if torch.cuda.is_available() else 'cpu',
        project='/work/runarto/runs/football',
        name='rbk_detector',
        verbose=True,
        patience=20,
        cos_lr=True,
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        cache=False
    )
    print("✅ Training complete. Model saved in:", results.save_dir)


if __name__ == '__main__':
    if prepare_data_structure():
        copy_images_and_labels(
            img_src='/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1',
            gt_txt='/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt',
            split='train'
        )
        copy_images_and_labels(
            img_src='/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/img1',
            gt_txt='/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/gt/gt.txt',
            split='val'
        )

    yaml_path = create_dataset_yaml()
    train_model(yaml_path, use_pretrained=False)


