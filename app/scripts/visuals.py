import cv2
import numpy as np
from PIL import Image
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_target(frame, target_position="center"):
    h, w, _ = frame.shape
    x_center = w // 2
    y_center = h // 2

    # Slightly inset targets from the frame edge (e.g., 5% in)
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.01)

    positions = {
        "top_left":      (margin_x, margin_y),
        "top_center":    (x_center, margin_y),
        "top_right":     (w - margin_x, margin_y),
        "center_left":   (margin_x, y_center),
        "center":        (x_center, y_center),
        "center_right":  (w - margin_x, y_center),
        "bottom_left":   (margin_x, h - margin_y),
        "bottom_center": (x_center, h - margin_y),
        "bottom_right":  (w - margin_x, h - margin_y),
    }

    if target_position not in positions:
        raise ValueError(
            "target_position must be one of: " +
            ", ".join(positions.keys())
        )

    x_target, y_target = positions[target_position]

    # Draw outer circle (white) for contrast, then green filled circle
    cv2.circle(frame, (x_target, y_target), radius=22, color=(255,255,255), thickness=5)
    cv2.circle(frame, (x_target, y_target), radius=15, color=(0, 255, 0), thickness=-1)

    return frame

def generate_heatmap(gaze_data, bg_image_file, bins=100, alpha=0.5):

    x = np.array([row['norm_x'] for row in gaze_data])
    y = np.array([row['norm_y'] for row in gaze_data])

    bg_image = Image.open(bg_image_file).convert('RGB')
    width, height = bg_image.size

    # If data is already in [0, width]/[0, height] range, skip normalization.
    # If data is roughly in [-1, 1], map to [0, width]/[0, height]
    if np.min(x) < 0 or np.max(x) > width:
        x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * width
    if np.min(y) < 0 or np.max(y) > height:
        y = ((y - np.min(y)) / (np.max(y) - np.min(y))) * height

    # Create 2D histogram heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins], range=[[0, width], [0, height]])

    # Plot image and overlay heatmap
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(bg_image, extent=[0, width, 0, height], aspect='auto', zorder=1)
    ax.imshow(
        heatmap.T,
        origin='lower',
        cmap='hot',
        interpolation='nearest',
        alpha=alpha,
        aspect='auto',
        extent=[0, width, 0, height],
        zorder=2,
    )
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Save bytes to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf).copy()
    buf.close()
    return result_img
