"""Gemini 305 - Single frame RGB-D capture"""
import os
import sys
from datetime import datetime
import cv2
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr

with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()
cfg = Config()
cfg.enable_stream(OBStreamType.COLOR_STREAM)
cfg.enable_stream(OBStreamType.DEPTH_STREAM)
if config['rgbd']['align']:
    cfg.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)
pipeline.start(cfg)

output_dir = os.path.join(config['output']['base_dir'], 'rgbd_single')
os.makedirs(output_dir, exist_ok=True)

print("Capturing RGB-D frame...")
frames = pipeline.wait_for_frames(1000)

if frames:
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if color_frame and depth_frame:
        color_img = frame_to_bgr(color_frame)
        depth_data = np.asanyarray(depth_frame.get_data()).reshape(depth_frame.get_height(), depth_frame.get_width())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{output_dir}/color_{timestamp}.png", color_img)
        cv2.imwrite(f"{output_dir}/depth_{timestamp}.png", depth_data)
        print(f"Saved to {output_dir}")
    else:
        print("Failed to get frames")
else:
    print("No frames received")

pipeline.stop()
