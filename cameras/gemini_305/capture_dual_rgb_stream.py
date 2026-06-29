"""Gemini 305 - Continuous dual RGB stream capture"""
import os
import sys
from datetime import datetime
import cv2
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr, has_dual_color_sensors, find_depth_work_mode

with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()
device = pipeline.get_device()
print(f"Device: {device.get_device_info().get_name()}")

# Switch to dual color mode if needed
if not has_dual_color_sensors(device):
    mode = find_depth_work_mode(device, "Dual Color Streams")
    if mode:
        device.set_depth_work_mode(mode)
        print("Switched to Dual Color mode")

cfg = Config()
cfg.enable_stream(OBSensorType.LEFT_COLOR_SENSOR)
cfg.enable_stream(OBSensorType.RIGHT_COLOR_SENSOR)
pipeline.start(cfg)

output_dir = os.path.join(config['output']['base_dir'], 'dual_rgb_stream', datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(output_dir, exist_ok=True)

print("Press SPACE to start/stop, Q to quit")
capturing = False
frame_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue

        left_frame = frames.get_frame_by_type(OBFrameType.LEFT_COLOR_FRAME)
        right_frame = frames.get_frame_by_type(OBFrameType.RIGHT_COLOR_FRAME)

        left_img = frame_to_bgr(left_frame.as_video_frame()) if left_frame else None
        right_img = frame_to_bgr(right_frame.as_video_frame()) if right_frame else None

        if left_img is None or right_img is None:
            continue

        if capturing:
            cv2.imwrite(f"{output_dir}/left_{frame_count:06d}.png", left_img)
            cv2.imwrite(f"{output_dir}/right_{frame_count:06d}.png", right_img)
            frame_count += 1

        display = np.hstack([left_img, right_img])
        cv2.putText(display, f"{'CAPTURING' if capturing else 'READY'} | {frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if capturing else (0, 255, 0), 2)
        cv2.imshow("Dual RGB", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            capturing = not capturing
        elif key in (ord('q'), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames")
