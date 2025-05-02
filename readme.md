# Datasyn Prosjekt Final

This project is focused on training and evaluating machine learning models for object detection and tracking in football videos. It includes scripts for data preparation, model training, evaluation, and video annotation. Developed for the course **TDT4265 – Computer Vision** at NTNU.

*Kok wisely.*

---

## 📁 Project Structure

datasyn-prosjekt-final/ ├── src/ # Core project modules │ ├── train_model.py # Script for training YOLO models │ ├── object_tracking.py # Object detection & pitch homography │ ├── match_processor.py # Match-wise orchestration & logic │ └── helpers/ # Utility functions (data conversion, etc.) │ ├── to_yolo_format.py # Converts annotation format │ └── ... # Other utility modules ├── main.py # Main entry point (for running pipeline) ├── requirements.txt # Python dependencies └── README.md # Project documentation


---

## ✨ Features

- **Model Training**: Train YOLO models for detecting footballs and players.
- **Object Tracking**: Detect and track objects in football matches using BoT-SORT.
- **Video Annotation**: Annotate video frames with bounding boxes and keypoints.
- **Data Preparation**: Convert raw datasets to YOLOv8 format.

---

## ⚙️ Setup

> ⚠️ This project assumes it is cloned into your `~/Documents/` folder.

You are expected to run this project on a machine in the **Cybele** lab at NTNU. You can either:

### 🔘 Use the machine physically  
Sit at any Cybele machine and launch a terminal.

### 🔗 Or SSH into a Cybele machine:

```bash
ssh your_username@clabXX.idi.ntnu.no

   cd datasyn-prosjekt-final