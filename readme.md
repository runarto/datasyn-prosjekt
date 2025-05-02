# Datasyn Prosjekt Final

This project is focused on training and evaluating machine learning models for object detection and tracking in football videos. It includes scripts for data preparation, model training, evaluation, and video annotation. Developed for the course **TDT4265 – Computer Vision** at NTNU.

*Kok wisely.*

---

## 📁 Project Structure

```text
datasyn-prosjekt-final/
├── src/                   # Core project modules
│   ├── train_model.py     # YOLO model training
│   ├── object_tracking.py # Object detection and tracking
│   ├── match_processor.py # Match orchestration and pipeline logic
│   └── helpers/           # Utility functions (data conversion, etc.)
│       ├── helpers.py     # Helper functions
├── main.py                # Main entry point to run the pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```
---

## Features

- **Model Training**: Train YOLO models for detecting footballs and players.
- **Object Tracking**: Detect and track objects in football matches using ByteTrack from Supervision.
- **Video Annotation**: Annotate video frames with markers and labels.
- **Data Preparation**: Convert raw datasets to YOLOv8 format.

---

## Setup

> ⚠️ This project assumes it is cloned into your `~/Documents/` folder.

1. Clone the repository:
   ```bash
   cd ~/Documents/
   git clone https://github.com/runarto/datasyn-prosjekt-final.git
   cd datasyn-prosjekt-final


2. Venv
   You can choose to create a venv
   ```bash
   python3 -m venv name-of-venv
   source name-of-venv/bin/activate

3. Regardless of whether or not you created a venv, install the requirements
   ```bash
   pip install -r requirements.txt

---

## How to Run

1. If this is your **first time running the project**, open `main.py` and set the `TRAIN` flag to `True` to train the detection model:
   ```python
   TRAIN = True

2. From the root directory of the project, do:
   ```bash
   python3 main.py
