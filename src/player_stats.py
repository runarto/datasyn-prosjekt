from collections import defaultdict
import cv2
import numpy as np
from mplsoccer import Pitch



class PlayerStats:
    def __init__(self, fps=30):
        self.tracks = defaultdict(list)  # {tracker_id: list of (frame_idx, (x, y))}
        self.fps = fps

        # Create pitch object once
        self.pitch = Pitch(pitch_type='custom', pitch_length=106.5, pitch_width=67,
                           line_color='white', pitch_color='green')

    def update(self, player_detections, homography, frame_idx):
        if homography is None:
            return

        xyxy = player_detections.xyxy
        tracker_ids = player_detections.tracker_id

        bottom_centers = np.stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            xyxy[:, 3]
        ], axis=1).astype(np.float32)

        field_coords = cv2.perspectiveTransform(bottom_centers[np.newaxis, ...], homography)[0]

        for idx, pos in zip(tracker_ids, field_coords):
            self.tracks[int(idx)].append((frame_idx, tuple(pos)))

    def compute_speed_acceleration(self, positions):
        if len(positions) < 2:
            return 0.0, 0.0

        frames, coords = zip(*positions)
        coords = np.array(coords)
        frames = np.array(frames)
        times = frames / self.fps

        dt = np.diff(times)
        dp = np.diff(coords, axis=0)
        speeds = np.linalg.norm(dp, axis=1) / dt

        if len(speeds) > 1:
            accels = np.diff(speeds) / dt[1:]
            return speeds[-1], accels[-1]
        else:
            return speeds[-1], 0.0

    def draw_2d_pitch_map(self, live=False, figsize=(10, 6)):
        import matplotlib.pyplot as plt

        if live:
            plt.clf()
            fig, ax = plt.gcf(), plt.gca()
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Draw pitch using mplsoccer
        self.pitch.draw(ax=ax)

        for player_id, track in self.tracks.items():
            if len(track) < 2:
                continue
            positions = [p for _, p in track]
            trajectory = np.array(positions)
            speed, acc = self.compute_speed_acceleration(track)
            label = f"#{player_id} | {speed:.2f} m/s | {acc:.2f} m/s²"
            ax.plot(trajectory[:, 0], trajectory[:, 1], label=label)

        if not live:
            ax.legend()
            plt.tight_layout()
            plt.show()
        else:
            plt.pause(0.001)
