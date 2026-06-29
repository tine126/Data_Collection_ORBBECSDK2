"""
Get camera intrinsics for Color, Depth, and IR streams.
Usage:
    conda activate basketball_env
    python get_camera_intrinsics.py
"""

import json
from pyorbbecsdk import *

def find_video_profile(profile_list, width, height, fps, fmt=None):
    """Find a matching video stream profile by resolution, fps, and optionally format."""
    count = profile_list.get_count()
    for i in range(count):
        p = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
        if p.get_width() == width and p.get_height() == height and p.get_fps() == fps:
            if fmt is None or p.get_format() == fmt:
                return p
    return None

def list_profiles(profile_list, sensor_name):
    """Print all available profiles for a sensor."""
    print(f"\n[{sensor_name}] Available profiles:")
    count = profile_list.get_count()
    for i in range(count):
        p = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
        print(f"  {p.get_width()}x{p.get_height()} @ {p.get_fps()}fps, format: {p.get_format()}")

def get_intrinsics():
    pipeline = Pipeline()
    config = Config()

    # Enable streams with current config resolution
    config.enable_video_stream(OBSensorType.COLOR_SENSOR, 1280, 720, 30, OBFormat.MJPG)
    config.enable_video_stream(OBSensorType.DEPTH_SENSOR, 1280, 720, 30, OBFormat.Y16)

    # Get IR profiles and list available options
    left_ir_profiles = pipeline.get_stream_profile_list(OBSensorType.LEFT_IR_SENSOR)
    right_ir_profiles = pipeline.get_stream_profile_list(OBSensorType.RIGHT_IR_SENSOR)

    list_profiles(left_ir_profiles, "LEFT_IR")
    list_profiles(right_ir_profiles, "RIGHT_IR")

    # Find 1280x720 profile (Y8 format, 60fps)
    left_ir_profile = find_video_profile(left_ir_profiles, 1280, 720, 30, OBFormat.Y8)
    right_ir_profile = find_video_profile(right_ir_profiles, 1280, 720, 30, OBFormat.Y8)

    if left_ir_profile is None:
        print("\n[WARNING] Left IR 1280x720@60 not found, using default")
        left_ir_profile = left_ir_profiles.get_default_video_stream_profile()
    if right_ir_profile is None:
        print("\n[WARNING] Right IR 1280x720@60 not found, using default")
        right_ir_profile = right_ir_profiles.get_default_video_stream_profile()

    config.enable_stream(left_ir_profile)
    config.enable_stream(right_ir_profile)

    pipeline.start(config)

    # Wait for frames to get stream profiles (IR may need several frames)
    frames = None
    for _ in range(10):
        frames = pipeline.wait_for_frames(1000)
        if frames:
            left_ir_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
            right_ir_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
            if left_ir_frame and right_ir_frame:
                print("Got all IR frames!")
                break
        print("Waiting for IR frames...")

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

        # Left IR
        left_ir_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
        if left_ir_frame:
            left_ir_profile = left_ir_frame.get_stream_profile().as_video_stream_profile()
            left_ir_intr = left_ir_profile.get_intrinsic()
            left_ir_dist = left_ir_profile.get_distortion()
            intrinsics["left_ir"] = {
                "width": left_ir_intr.width,
                "height": left_ir_intr.height,
                "fx": left_ir_intr.fx,
                "fy": left_ir_intr.fy,
                "cx": left_ir_intr.cx,
                "cy": left_ir_intr.cy,
                "k1": left_ir_dist.k1, "k2": left_ir_dist.k2, "k3": left_ir_dist.k3,
                "k4": left_ir_dist.k4, "k5": left_ir_dist.k5, "k6": left_ir_dist.k6,
                "p1": left_ir_dist.p1, "p2": left_ir_dist.p2,
            }

        # Right IR
        right_ir_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
        if right_ir_frame:
            right_ir_profile = right_ir_frame.get_stream_profile().as_video_stream_profile()
            right_ir_intr = right_ir_profile.get_intrinsic()
            right_ir_dist = right_ir_profile.get_distortion()
            intrinsics["right_ir"] = {
                "width": right_ir_intr.width,
                "height": right_ir_intr.height,
                "fx": right_ir_intr.fx,
                "fy": right_ir_intr.fy,
                "cx": right_ir_intr.cx,
                "cy": right_ir_intr.cy,
                "k1": right_ir_dist.k1, "k2": right_ir_dist.k2, "k3": right_ir_dist.k3,
                "k4": right_ir_dist.k4, "k5": right_ir_dist.k5, "k6": right_ir_dist.k6,
                "p1": right_ir_dist.p1, "p2": right_ir_dist.p2,
            }

        # Extrinsics: color -> depth
        if color_frame and depth_frame:
            extr = color_profile.get_extrinsic_to(depth_profile)
            intrinsics["extrinsic_color_to_depth"] = {
                "rotation": extr.rot.tolist(),       # 3x3 matrix
                "translation": extr.transform.tolist() # 3 floats, mm
            }

        # Extrinsics: left_ir -> right_ir (baseline)
        if left_ir_frame and right_ir_frame:
            extr_ir = left_ir_profile.get_extrinsic_to(right_ir_profile)
            intrinsics["extrinsic_left_ir_to_right_ir"] = {
                "rotation": extr_ir.rot.tolist(),
                "translation": extr_ir.transform.tolist()
            }
            print(f"\nLeft-Right IR Baseline: {extr_ir.transform[0]:.2f} mm")

    pipeline.stop()

    # Save to JSON
    with open("camera_intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)

    print("Camera Intrinsics & Extrinsics:")
    print(json.dumps(intrinsics, indent=2))
    print("\nSaved to camera_intrinsics.json")

if __name__ == "__main__":
    get_intrinsics()
