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
from helpers.helper import to_yolo_format

IDUN = True
if IDUN:
    # If running on IDUN, set the SRC_DIR to the appropriate path
    SRC_DIR = os.path.join(os.path.dirname(__file__), "data")
    BASE_DIR = '/cluster/projects/vc/data/other/open/RBK_TDT17'
else:
    SRC_DIR = "/work/runarto/data"
    BASE_DIR = '/datasets/tdt4265/other/rbk/'

TRAIN_DIR = [
    os.path.join(BASE_DIR, '1_train-val_1min_aalesund_from_start'),
    os.path.join(BASE_DIR, '2_train-val_1min_after_goal')
]

VAL_DIR = [os.path.join(BASE_DIR, '2_train-val_1min_after_goal')]
TEST_DIR = [os.path.join(BASE_DIR, '3_test_1min_hamkam_from_start')]
FINAL_VAL_DIR = os.path.join(BASE_DIR, '4_annotate_1min_bodo_start')


class Train:
    def __init__(self):
        self.crop_dir = os.path.join(SRC_DIR, 'ball_crops')

    def create_dataset_yaml(self):
        yaml_path = os.path.join(SRC_DIR, "rbk_yaml.yaml")
        abs_dataset_path = os.path.abspath(SRC_DIR)
        config = {
            'path': abs_dataset_path,
            'train': os.path.join(abs_dataset_path, 'images/train'),
            'test': os.path.join(abs_dataset_path, 'images/test'),
            'val': os.path.join(abs_dataset_path, 'images/val'),
            'nc': 2,
            'names': ['ball', 'player']
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        print(f"✅ Created dataset config at {yaml_path}")
        return yaml_path

    def prepare_data_structrue(self):
        if os.path.exists(SRC_DIR):  
            shutil.rmtree(SRC_DIR) # Remove existing directory to avoid duplicates
            os.makedirs(SRC_DIR, exist_ok=True) 

        # Create the new directory structure
        os.makedirs(os.path.join(SRC_DIR, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, 'images/val'), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, 'images/test'), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, 'labels/train'), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, 'labels/val'), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, 'labels/test'), exist_ok=True)

        # Copy images and labels to new structure
        index = 1
        for dataset_dir in TRAIN_DIR:
            img_dir = os.path.join(dataset_dir, 'img1')
            label_dir = os.path.join(dataset_dir, 'gt')

            for img_path in glob(f"{img_dir}/*.jpg"):
                filename = os.path.basename(img_path)
                # New filename is index_xxxxx, where the xxxxx is the original filename without extension
                filename = f'{index}_{filename}'
                shutil.copy(img_path, os.path.join(SRC_DIR, 'images/train', filename))

            for label_path in glob(f"{label_dir}/*.txt"):
                filename = os.path.basename(label_path)
                filename = f'{index}_{filename}'
                shutil.copy(label_path, os.path.join(SRC_DIR, 'labels/train', filename))

            index += 1


        for dataset_dir in VAL_DIR:
            img_dir = os.path.join(dataset_dir, 'img1')
            label_dir = os.path.join(dataset_dir, 'gt')

            for img_path in glob(f"{img_dir}/*.jpg"):
                shutil.copy(img_path, os.path.join(SRC_DIR, 'images/val'))

            for label_path in glob(f"{label_dir}/*.txt"):
                shutil.copy(label_path, os.path.join(SRC_DIR, 'labels/val'))
            
        for dataset_dir in TEST_DIR:
            img_dir = os.path.join(dataset_dir, 'img1')
            label_dir = os.path.join(dataset_dir, 'gt')

            print(f"Directory exists? {os.path.exists(img_dir)}")
            for img_path in glob(f"{img_dir}/*.jpg"):
                shutil.copy(img_path, os.path.join(SRC_DIR, 'images/test'))

            for label_path in glob(f"{label_dir}/*.txt"):
                shutil.copy(label_path, os.path.join(SRC_DIR, 'labels/test'))

        to_yolo_format(SRC_DIR, 'train')
        to_yolo_format(SRC_DIR, 'val')
        to_yolo_format(SRC_DIR, 'test')

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


    def train_model(self):
                # Start from COCO-pretrained YOLOv8m
        data_yaml = self.create_dataset_yaml()
        device = '0' if torch.cuda.is_available() else 'cpu'
        model = YOLO('yolov8x.pt')

        model.train(
            data=data_yaml,
            epochs=80,                   # ample epochs to master ball
            batch=6,                     # fit in GPU memory at 1280px
            imgsz=1280,                  # high resolution for small ball
            device=device,
            project='ball_player_model', # base folder
            name='phase1_ball',          # sub-folder for this run
            exist_ok=True,               # reuse if already exists
            classes=[0],                 # only the ball class
            lr0=0.001,                   # lower than default to fine-tune
            cos_lr=True,                 # cosine decay schedule
            warmup_epochs=5,             # ramp LR over first few epochs
            patience=15,                 # early stop if no gain for 15 epochs
            save_period=5,               # checkpoint every 5 epochs
            cache=True,                  # speed up I/O
            # realistic “football” augmentations
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5, fliplr=0.5,
            mosaic=1.0, mixup=0.0,       # heavy mosaic early, no mixup
        )

        # Load the best ball-only weights
        best_ball = 'runs/ball_player_model/phase1_ball/weights/best.pt'
        model = YOLO(best_ball)

        # ------------------------------------------------------------------------------
        # Phase 2: Multi-class fine-tuning (ball + players)
        # ------------------------------------------------------------------------------

        model.train(
            data=data_yaml,
            epochs=120,                  # more epochs for full task
            batch=8,
            imgsz=1280,
            device=device,
            project='ball_player_model',
            name='phase2_full',
            exist_ok=True,
            classes=[0, 1],              # ball + player
            lr0=0.0005,                  # lower LR to avoid forgetting ball
            cos_lr=True,
            warmup_epochs=2,             # shorter warmup when fine-tuning
            patience=25,                 # allow more patience now
            save_period=5,
            cache=True,
            # keep same realistic augmentations, but turn off mosaic
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            translate=0.1, scale=0.5, fliplr=0.5,
            mosaic=0.0, mixup=0.0,
        )


    def evaluate_model(self, weights):
        data_yaml = self.create_dataset_yaml()
        model = YOLO(weights)
        results = model.val(
            data=data_yaml,  # path to your dataset config
            split="test",          # test/val/train
            imgsz=640,             # image size used during validation
            conf=0.001,            # confidence threshold (default: 0.001)
            iou=0.6,               # IoU threshold for mAP (default: 0.6)
            plots=True,            # save precision-recall curves, confusion matrix etc.
            save_json=True,        # export COCO-format predictions
            save_hybrid=False,     # save hybrid labels (labels + predictions)
            half=True,             # use half precision (if supported)
            device=0               # set device, e.g., 'cpu', 0, 1
        )
        print("✅ Evaluation complete. Results saved in:", results.save_dir)



if __name__ == "__main__":

    train = Train()
    weights = "ball_player_model/best.pt"
    train.prepare_data_structrue()
    train.evaluate_model(weights)

    