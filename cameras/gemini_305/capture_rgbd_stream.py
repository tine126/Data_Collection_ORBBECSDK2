"""Gemini 305 - Continuous RGB-D stream capture"""
import os
import sys
from datetime import datetime
import cv2
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr, visualize_depth

with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), encoding='utf-8') as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()
cfg = Config()
cfg.enable_stream(OBStreamType.COLOR_STREAM)
cfg.enable_stream(OBStreamType.DEPTH_STREAM)
if config['rgbd']['align']:
    cfg.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)
pipeline.start(cfg)

output_dir = os.path.join(config['output']['base_dir'], 'rgbd_stream', datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(output_dir, exist_ok=True)

print("Press SPACE to start/stop, Q to quit")
capturing = False
frame_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_img = frame_to_bgr(color_frame) if color_frame else None
        depth_data = np.asanyarray(depth_frame.get_data()).reshape(depth_frame.get_height(), depth_frame.get_width()) if depth_frame else None

        if color_img is None or depth_data is None:
            continue

        if capturing:
            cv2.imwrite(f"{output_dir}/color_{frame_count:06d}.png", color_img)
            cv2.imwrite(f"{output_dir}/depth_{frame_count:06d}.png", depth_data)
            frame_count += 1

        depth_vis = visualize_depth(depth_data, config['rgbd']['min_depth_mm'], config['rgbd']['max_depth_mm'])
        display = np.hstack([color_img, cv2.resize(depth_vis, (color_img.shape[1], color_img.shape[0]))])
        cv2.putText(display, f"{'CAPTURING' if capturing else 'READY'} | {frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if capturing else (0, 255, 0), 2)
        cv2.imshow("RGB-D", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            capturing = not capturing
        elif key in (ord('q'), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames")
