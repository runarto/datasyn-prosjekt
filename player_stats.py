from collections import defaultdict, deque
import cv2
import numpy as np
from mplsoccer import Pitch
import math
import matplotlib.pyplot as plt

class PlayerStats:
    def __init__(self, fps=30, frame_gap=30, pitch_length=106.5, pitch_width=67, pitch_image_width=1920):
        self.tracks = defaultdict(lambda: deque(maxlen=frame_gap + 1))  # Limited queue for each player
        self.fps = fps
        self.frame_gap = frame_gap
        self.pitch_length = pitch_length  # in meters
        self.pitch_width = pitch_width    # in meters
        self.pitch_image_width = pitch_image_width  # pixels (assuming homography is normalized to this)
        
        # Create pitch object (for visualization if needed)
        self.pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
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

    def compute_player_stats(self, tracker_id):
        positions = list(self.tracks[tracker_id])
        if len(positions) < 2:
            return 0.0, 0.0, 0.0

        pos0_frame, pos0 = positions[0]
        pos1_frame, pos1 = positions[-1]

        # Δpixels (Equation 3.5)
        dx_pixels = pos1[0] - pos0[0]
        dy_pixels = pos1[1] - pos0[1]
        delta_pixels = np.sqrt(dx_pixels**2 + dy_pixels**2)

        # Δmeters (Equation 3.6)
        pitch_pixels = self.pitch_image_width
        pitch_meters = self.pitch_length  # or self.pitch_width depending on direction
        delta_meters = delta_pixels * (pitch_meters / pitch_pixels)

        # Speed (Equation 3.7 and 3.8)
        frame_diff = pos1_frame - pos0_frame
        if frame_diff == 0:
            speed_m_s = 0.0
        else:
            speed_m_s = delta_meters * (frame_diff / self.frame_gap)

        # Direction (Equation 3.9)
        direction_rad = math.atan2(dy_pixels, dx_pixels)

        return delta_meters, speed_m_s, direction_rad

    def compute_all_stats(self):
        results = {}
        for tracker_id in self.tracks.keys():
            distance, speed, direction = self.compute_player_stats(tracker_id)
            results[tracker_id] = {
                'distance_m': distance,
                'speed_m_s': speed,
                'direction_rad': direction
            }
        return results
    

    def draw_2d_pitch_map(self, live=True):
        stats = self.compute_all_stats()
        fig, ax = self.pitch.draw(figsize=(10, 7))

        for tracker_id, stat in stats.items():
            positions = list(self.tracks[tracker_id])
            if len(positions) < 1:
                continue
            _, latest_pos = positions[-1]

            # Draw player circle with ID inside
            circle = plt.Circle(
                (latest_pos[0], latest_pos[1]), 
                radius=1.5, 
                edgecolor='black', 
                facecolor='white', 
                linewidth=1.5, 
                zorder=5
            )
            ax.add_patch(circle)
            ax.text(
                latest_pos[0], latest_pos[1], str(tracker_id),
                ha='center', va='center', fontsize=7, color='black', zorder=6
            )

            # Draw movement direction triangle
            direction = stat['direction_rad']
            dx = np.cos(direction)
            dy = np.sin(direction)

            # Triangle tip position
            tip_x = latest_pos[0] + dx * 3
            tip_y = latest_pos[1] + dy * 3

            # Base of triangle (small width perpendicular to direction)
            left_x = latest_pos[0] + np.cos(direction + np.pi/2) * 1
            left_y = latest_pos[1] + np.sin(direction + np.pi/2) * 1
            right_x = latest_pos[0] + np.cos(direction - np.pi/2) * 1
            right_y = latest_pos[1] + np.sin(direction - np.pi/2) * 1

            triangle = plt.Polygon(
                [[tip_x, tip_y], [left_x, left_y], [right_x, right_y]],
                color='red', zorder=7
            )
            ax.add_patch(triangle)

        # OPTIONAL: If you have ball position tracking
        if hasattr(self, 'ball_tracks') and self.ball_tracks:
            ball_positions = list(self.ball_tracks)
            if ball_positions:
                _, ball_pos = ball_positions[-1]
                ax.scatter(ball_pos[0], ball_pos[1], color='orange', s=50, label='Ball', zorder=8)

        if not live:
            plt.title('Final 2D Map of Players and Ball')
        else:
            plt.title('Live 2D Map')

        plt.legend()
        plt.show()
