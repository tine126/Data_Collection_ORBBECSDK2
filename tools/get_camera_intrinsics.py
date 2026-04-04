"""
Get camera intrinsics for Color, Depth, and IR streams.
Usage:
    conda activate basketball_env
    python get_camera_intrinsics.py
"""

import json
from pyorbbecsdk import *

def get_intrinsics():
    pipeline = Pipeline()
    config = Config()

    # Enable streams with current config resolution
    config.enable_video_stream(OBSensorType.COLOR_SENSOR, 640, 480, 30, OBFormat.MJPG)
    config.enable_video_stream(OBSensorType.DEPTH_SENSOR, 640, 480, 30, OBFormat.Y16)

    pipeline.start(config)

    # Wait for frames to get stream profiles
    frames = pipeline.wait_for_frames(1000)

    intrinsics = {}

    if frames:
        color_frame = frames.get_color_frame()
        if color_frame:
            color_profile = color_frame.get_stream_profile().as_video_stream_profile()
            color_intr = color_profile.get_intrinsic()
            color_dist = color_profile.get_distortion()
            intrinsics["color"] = {
                "width": color_intr.width,
                "height": color_intr.height,
                "fx": color_intr.fx,
                "fy": color_intr.fy,
                "cx": color_intr.cx,
                "cy": color_intr.cy,
                "k1": color_dist.k1, "k2": color_dist.k2, "k3": color_dist.k3,
                "k4": color_dist.k4, "k5": color_dist.k5, "k6": color_dist.k6,
                "p1": color_dist.p1, "p2": color_dist.p2,
            }

        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_profile = depth_frame.get_stream_profile().as_video_stream_profile()
            depth_intr = depth_profile.get_intrinsic()
            depth_dist = depth_profile.get_distortion()
            intrinsics["depth"] = {
                "width": depth_intr.width,
                "height": depth_intr.height,
                "fx": depth_intr.fx,
                "fy": depth_intr.fy,
                "cx": depth_intr.cx,
                "cy": depth_intr.cy,
                "k1": depth_dist.k1, "k2": depth_dist.k2, "k3": depth_dist.k3,
                "k4": depth_dist.k4, "k5": depth_dist.k5, "k6": depth_dist.k6,
                "p1": depth_dist.p1, "p2": depth_dist.p2,
            }

        # Extrinsics: color -> depth
        if color_frame and depth_frame:
            extr = color_profile.get_extrinsic_to(depth_profile)
            intrinsics["extrinsic_color_to_depth"] = {
                "rotation": extr.rot.tolist(),       # 3x3 matrix
                "translation": extr.transform.tolist() # 3 floats, mm
            }

    pipeline.stop()

    # Save to JSON
    with open("camera_intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)

    print("Camera Intrinsics & Extrinsics:")
    print(json.dumps(intrinsics, indent=2))
    print("\nSaved to camera_intrinsics.json")

if __name__ == "__main__":
    get_intrinsics()
