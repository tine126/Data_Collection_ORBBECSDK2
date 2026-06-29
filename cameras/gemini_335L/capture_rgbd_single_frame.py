"""Gemini 335L - Interactive RGB-D single frame capture

Press SPACE to capture a frame (RGB + Depth)
Press Q to quit
"""
import os
import sys
from datetime import datetime
import cv2
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr

# Load config
with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("Initializing Gemini 335L...")
pipeline = Pipeline()
device = pipeline.get_device()
print(f"Device: {device.get_device_info().get_name()}")

# Configure color + depth streams
cfg = Config()
color_cfg = config['rgbd']['color']
depth_cfg = config['rgbd']['depth']

cfg.enable_video_stream(
    OBStreamType.COLOR_STREAM,
    color_cfg['width'], color_cfg['height'], color_cfg['fps'],
    OBFormat.MJPG
)
cfg.enable_video_stream(
    OBStreamType.DEPTH_STREAM,
    depth_cfg['width'], depth_cfg['height'], depth_cfg['fps'],
    OBFormat.Y16
)

# Always enable depth-to-color alignment
cfg.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)
print("Depth-to-Color alignment: ON")

pipeline.start(cfg)

# Create output directory
output_dir = os.path.join(config['output']['base_dir'], 'rgbd_single_frame')
os.makedirs(output_dir, exist_ok=True)

print("\n[Controls]")
print("  SPACE - Capture frame")
print("  Q/ESC - Quit")
print("\nReady. Showing preview...")

frame_count = 0
colormap = cv2.COLORMAP_JET

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames
        color_img = frame_to_bgr(color_frame)
        if color_img is None:
            continue

        depth_data = np.asanyarray(depth_frame.get_data()).reshape(
            depth_frame.get_height(),
            depth_frame.get_width()
        )

        # Create depth visualization for preview (colormap)
        depth_clipped = np.clip(
            depth_data,
            config['rgbd']['min_depth_mm'],
            config['rgbd']['max_depth_mm']
        )
        depth_normalized = (
            (depth_clipped - config['rgbd']['min_depth_mm']) /
            (config['rgbd']['max_depth_mm'] - config['rgbd']['min_depth_mm']) * 255
        ).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_normalized, colormap)

        # Resize depth to match color for display
        depth_vis_resized = cv2.resize(depth_vis, (color_img.shape[1], color_img.shape[0]))

        # Display side by side
        display = np.hstack([color_img, depth_vis_resized])
        cv2.putText(display, f"Frames captured: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=Capture | Q=Quit", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("RGB-D Preview (Press SPACE to capture)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Capture frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save RGB
            rgb_path = f"{output_dir}/rgb_{timestamp}.png"
            cv2.imwrite(rgb_path, color_img)

            # Save raw depth (16-bit PNG)
            depth_path = f"{output_dir}/depth_{timestamp}.png"
            cv2.imwrite(depth_path, depth_data)

            frame_count += 1
            print(f"✓ Frame {frame_count} saved: rgb_{timestamp}.png + depth_{timestamp}.png")

        elif key in (ord('q'), 27):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\n✅ Total frames captured: {frame_count}")
    print(f"   Output directory: {output_dir}")

