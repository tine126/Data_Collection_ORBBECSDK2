"""Gemini 335L - Get camera parameters (intrinsics/extrinsics)"""
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

# Get color camera intrinsics
try:
    cfg = Config()
    cfg.enable_stream(OBStreamType.COLOR_STREAM)
    pipeline.start(cfg)

    frames = pipeline.wait_for_frames(1000)
    if frames:
        color_frame = frames.get_color_frame()
        if color_frame:
            profile = color_frame.get_stream_profile()
            intrinsics = profile.get_intrinsics()
            params['color'] = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.cx,
                'cy': intrinsics.cy,
                'distortion': list(intrinsics.distortion)
            }
    pipeline.stop()
except Exception as e:
    print(f"Color intrinsics failed: {e}")

# Get depth camera intrinsics
try:
    cfg = Config()
    cfg.enable_stream(OBStreamType.DEPTH_STREAM)
    pipeline.start(cfg)

    frames = pipeline.wait_for_frames(1000)
    if frames:
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            profile = depth_frame.get_stream_profile()
            intrinsics = profile.get_intrinsics()
            params['depth'] = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.cx,
                'cy': intrinsics.cy,
                'distortion': list(intrinsics.distortion)
            }
    pipeline.stop()
except Exception as e:
    print(f"Depth intrinsics failed: {e}")

output_path = os.path.join(os.path.dirname(__file__), 'camera_params.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(params, f, indent=2)

print(f"\nParameters saved to: {output_path}")
print(json.dumps(params, indent=2))
