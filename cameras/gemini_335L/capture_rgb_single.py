"""Gemini 335L - Single frame RGB capture"""
import os
import sys
from datetime import datetime
import cv2
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr

with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()
cfg = Config()
cfg.enable_stream(OBSensorType.COLOR_SENSOR)
pipeline.start(cfg)

output_dir = os.path.join(config['output']['base_dir'], 'rgb_single')
os.makedirs(output_dir, exist_ok=True)

print("Capturing single RGB frame...")
frames = pipeline.wait_for_frames(1000)
color_frame = frames.get_color_frame() if frames else None

if color_frame:
    img = frame_to_bgr(color_frame)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/rgb_{timestamp}.png"
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
else:
    print("Failed to capture frame")

pipeline.stop()
