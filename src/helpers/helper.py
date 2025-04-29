import os
from glob import glob

def to_yolo_format(SRC_DIR, split, img_size=(1920, 1080)):
    """
    Converts MOT-style *_gt.txt to YOLO .txt labels (per frame).
    Handles both indexed filenames (e.g., 1_gt.txt) and raw (gt.txt).
    """
    gt_dir = os.path.join(SRC_DIR, 'labels', split)
    label_output_dir = os.path.join(SRC_DIR, 'labels', split)
    os.makedirs(label_output_dir, exist_ok=True)

    gt_files = glob(os.path.join(gt_dir, '*.txt'))
    if not gt_files:
        print(f"⚠️ No gt.txt files found in {gt_dir}")
        return

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        if "gt" not in filename: # Skip any files that are not *_gt.txt
            continue
        index_prefix = ""

        # If filename starts with number_, extract it
        if '_' in filename and filename.split('_')[0].isdigit():
            index_prefix = f"{filename.split('_')[0]}_"

        frame_to_lines = {}

        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9 or float(parts[8]) == 0.0:
                    continue

                frame, _, x, y, w, h, _, cls_id, _ = parts
                frame = int(frame)
                x, y, w, h = map(float, (x, y, w, h))
                img_w, img_h = img_size

                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                yolo_cls = 0 if int(cls_id) == 1 else 1
                yolo_line = f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

                frame_key = f"{index_prefix}{frame:06d}"
                frame_to_lines.setdefault(frame_key, []).append(yolo_line)

        # Write YOLO .txt files per frame
        for frame_key, yolo_lines in frame_to_lines.items():
            out_path = os.path.join(label_output_dir, f"{frame_key}.txt")
            with open(out_path, 'w') as out_file:
                out_file.write('\n'.join(yolo_lines) + '\n')

        print(f"✅ Converted: {filename} → {len(frame_to_lines)} labels written.")


def estimate_energy_and_distance(compute_seconds, gpu_power_watts=450, tesla_efficiency_kwh_per_100km=17):
    hours = compute_seconds / 3600
    energy_kwh = (gpu_power_watts * hours) / 1000  # Watts × hours → kWh
    distance_km = (energy_kwh / tesla_efficiency_kwh_per_100km) * 100
    return energy_kwh, distance_km