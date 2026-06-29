"""Gemini 305 - Single frame dual RGB capture"""
import os
import sys
from datetime import datetime
import cv2
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr, has_dual_color_sensors, find_depth_work_mode

with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), encoding='utf-8') as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()
device = pipeline.get_device()

if not has_dual_color_sensors(device):
    mode = find_depth_work_mode(device, "Dual Color Streams")
    if mode:
        device.set_depth_work_mode(mode)

cfg = Config()
cfg.enable_stream(OBSensorType.LEFT_COLOR_SENSOR)
cfg.enable_stream(OBSensorType.RIGHT_COLOR_SENSOR)
pipeline.start(cfg)

output_dir = os.path.join(config['output']['base_dir'], 'dual_rgb_single')
os.makedirs(output_dir, exist_ok=True)

print("Capturing dual RGB frame...")
frames = pipeline.wait_for_frames(1000)

if frames:
    left_frame = frames.get_frame_by_type(OBFrameType.LEFT_COLOR_FRAME)
    right_frame = frames.get_frame_by_type(OBFrameType.RIGHT_COLOR_FRAME)

    if left_frame and right_frame:
        left_img = frame_to_bgr(left_frame.as_video_frame())
        right_img = frame_to_bgr(right_frame.as_video_frame())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{output_dir}/left_{timestamp}.png", left_img)
        cv2.imwrite(f"{output_dir}/right_{timestamp}.png", right_img)
        print(f"Saved to {output_dir}")
    else:
        print("Failed to get frames")
else:
    print("No frames received")

pipeline.stop()
