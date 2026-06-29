"""Gemini 335L - Continuous RGB stream capture"""
import os
import sys
from datetime import datetime
import cv2
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *
from cameras.shared import frame_to_bgr

# Load config
with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    config = yaml.safe_load(f)

pipeline = Pipeline()
device = pipeline.get_device()
print(f"Device: {device.get_device_info().get_name()}")

cfg = Config()
cfg.enable_stream(OBSensorType.COLOR_SENSOR)
pipeline.start(cfg)

output_dir = os.path.join(config['output']['base_dir'], 'rgb_stream', datetime.now().strftime("%Y%m%d_%H%M%S"))
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
        if not color_frame:
            continue

        img = frame_to_bgr(color_frame)
        if img is None:
            continue

        if capturing:
            cv2.imwrite(f"{output_dir}/frame_{frame_count:06d}.png", img)
            frame_count += 1

        cv2.putText(img, f"{'CAPTURING' if capturing else 'READY'} | {frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if capturing else (0, 255, 0), 2)
        cv2.imshow("RGB Stream", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            capturing = not capturing
            print(f"{'Started' if capturing else 'Stopped'} | Frames: {frame_count}")
        elif key in (ord('q'), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames to {output_dir}")
