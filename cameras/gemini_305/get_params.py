"""Gemini 305 - Get camera parameters (intrinsics/extrinsics)"""
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *

pipeline = Pipeline()
device = pipeline.get_device()
info = device.get_device_info()
print(f"Device: {info.get_name()}")

params = {}

# Color intrinsics
try:
    cfg = Config()
    cfg.enable_stream(OBStreamType.COLOR_STREAM)
    pipeline.start(cfg)
    frames = pipeline.wait_for_frames(1000)
    if frames:
        color_frame = frames.get_color_frame()
        if color_frame:
            profile = color_frame.get_stream_profile()
            intr = profile.get_intrinsics()
            params['color'] = {
                'width': intr.width, 'height': intr.height,
                'fx': intr.fx, 'fy': intr.fy,
                'cx': intr.cx, 'cy': intr.cy,
                'distortion': list(intr.distortion)
            }
    pipeline.stop()
except Exception as e:
    print(f"Color failed: {e}")

# Depth intrinsics
try:
    cfg = Config()
    cfg.enable_stream(OBStreamType.DEPTH_STREAM)
    pipeline.start(cfg)
    frames = pipeline.wait_for_frames(1000)
    if frames:
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            profile = depth_frame.get_stream_profile()
            intr = profile.get_intrinsics()
            params['depth'] = {
                'width': intr.width, 'height': intr.height,
                'fx': intr.fx, 'fy': intr.fy,
                'cx': intr.cx, 'cy': intr.cy,
                'distortion': list(intr.distortion)
            }
    pipeline.stop()
except Exception as e:
    print(f"Depth failed: {e}")

output_path = os.path.join(os.path.dirname(__file__), 'camera_params.json')
with open(output_path, 'w') as f:
    json.dump(params, f, indent=2)

print(f"\nSaved to: {output_path}")
print(json.dumps(params, indent=2))
