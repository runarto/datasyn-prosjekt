from ultralytics import YOLO


model = YOLO('yolov8s.pt')  # Load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.train(
        data="your_dataset/rbk.yaml",
        epochs=100,
        patience=5,
        imgsz=1088,
        batch=4
    )
