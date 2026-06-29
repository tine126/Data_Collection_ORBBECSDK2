import json
import os
import math

from pyorbbecsdk import *


DUAL_COLOR_MODE_NAME = "Dual Color Streams"
WIDTH = 1280
HEIGHT = 800
FPS = 30
FORMAT = OBFormat.BGR


def intrinsic_to_dict(intrinsic):
    return {
        "width": intrinsic.width,
        "height": intrinsic.height,
        "fx": intrinsic.fx,
        "fy": intrinsic.fy,
        "cx": intrinsic.cx,
        "cy": intrinsic.cy,
    }


def distortion_to_dict(distortion):
    return {
        "k1": distortion.k1,
        "k2": distortion.k2,
        "k3": distortion.k3,
        "k4": distortion.k4,
        "k5": distortion.k5,
        "k6": distortion.k6,
        "p1": distortion.p1,
        "p2": distortion.p2,
    }


def extrinsic_to_dict(extrinsic):
    return {
        "rotation": extrinsic.rot.tolist(),
        "translation_mm": extrinsic.transform.tolist(),
    }


def find_depth_work_mode(device, target_name):
    mode_list = device.get_depth_work_mode_list()
    for i in range(mode_list.get_count()):
        mode = mode_list.get_depth_work_mode_by_index(i)
        if str(mode) == target_name:
            return mode
    return None


def ensure_dual_color_mode(device):
    original_mode = device.get_depth_work_mode()
    if str(original_mode) == DUAL_COLOR_MODE_NAME:
        return original_mode, False

    dual_color_mode = find_depth_work_mode(device, DUAL_COLOR_MODE_NAME)
    if dual_color_mode is None:
        raise RuntimeError(f"Device does not support {DUAL_COLOR_MODE_NAME}.")

    print(f"Switching depth work mode: {original_mode} -> {dual_color_mode}")
    device.set_depth_work_mode(dual_color_mode)
    return original_mode, True


def find_video_profile(profile_list, width, height, fps, fmt):
    for i in range(profile_list.get_count()):
        profile = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
        if (
            profile.get_width() == width
            and profile.get_height() == height
            and profile.get_fps() == fps
            and profile.get_format() == fmt
        ):
            return profile
    raise RuntimeError(f"Profile not found: {width}x{height}@{fps} {fmt}")


def wait_for_dual_rgb_frames(pipeline, timeout_ms=1000, max_attempts=20):
    for _ in range(max_attempts):
        frames = pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            continue

        left_frame = frames.get_frame_by_type(OBFrameType.LEFT_COLOR_FRAME)
        right_frame = frames.get_frame_by_type(OBFrameType.RIGHT_COLOR_FRAME)
        if left_frame and right_frame:
            return left_frame, right_frame

    raise RuntimeError("Timed out waiting for synchronized left/right RGB frames.")


def find_matching_calibration(device, width, height):
    param_list = device.get_calibration_camera_param_list()
    for i in range(param_list.get_count()):
        camera_param = param_list.get_camera_param(i)
        rgb_intrinsic = camera_param.rgb_intrinsic
        if rgb_intrinsic.width == width and rgb_intrinsic.height == height:
            return camera_param
    return None


def main():
    pipeline = Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()

    print(f"Device: {info.get_name()}")
    print(f"Serial: {info.get_serial_number()}")

    original_mode = None
    mode_switched = False

    try:
        original_mode, mode_switched = ensure_dual_color_mode(device)

        left_profiles = pipeline.get_stream_profile_list(OBSensorType.LEFT_COLOR_SENSOR)
        right_profiles = pipeline.get_stream_profile_list(OBSensorType.RIGHT_COLOR_SENSOR)
        left_profile = find_video_profile(left_profiles, WIDTH, HEIGHT, FPS, FORMAT)
        right_profile = find_video_profile(right_profiles, WIDTH, HEIGHT, FPS, FORMAT)

        config = Config()
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        config.enable_stream(left_profile)
        config.enable_stream(right_profile)

        print(f"Starting Dual RGB pipeline at {WIDTH}x{HEIGHT}@{FPS} {FORMAT} ...")
        pipeline.start(config)

        left_frame, right_frame = wait_for_dual_rgb_frames(pipeline)
        left_stream_profile = left_frame.get_stream_profile().as_video_stream_profile()
        right_stream_profile = right_frame.get_stream_profile().as_video_stream_profile()

        calibration = find_matching_calibration(device, WIDTH, HEIGHT)

        result = {
            "device": {
                "name": info.get_name(),
                "serial_number": info.get_serial_number(),
            },
            "mode": str(device.get_depth_work_mode()),
            "stream": {
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "format": str(FORMAT),
            },
            "left_rgb": {
                "intrinsic": intrinsic_to_dict(left_stream_profile.get_intrinsic()),
                "distortion": distortion_to_dict(left_stream_profile.get_distortion()),
            },
            "right_rgb": {
                "intrinsic": intrinsic_to_dict(right_stream_profile.get_intrinsic()),
                "distortion": distortion_to_dict(right_stream_profile.get_distortion()),
            },
        }

        if calibration is not None:
            translation = calibration.transform.transform.tolist()
            baseline_x_mm = abs(translation[0])
            baseline_mm = math.sqrt(sum(value * value for value in translation))
            result["calibration_reference"] = {
                "source": "device calibration table",
                "matched_rgb_resolution": [calibration.rgb_intrinsic.width, calibration.rgb_intrinsic.height],
                "entry_rgb_intrinsic": intrinsic_to_dict(calibration.rgb_intrinsic),
                "entry_rgb_distortion": distortion_to_dict(calibration.rgb_distortion),
                "entry_depth_intrinsic": intrinsic_to_dict(calibration.depth_intrinsic),
                "entry_depth_distortion": distortion_to_dict(calibration.depth_distortion),
                "entry_transform": extrinsic_to_dict(calibration.transform),
                "baseline_x_mm": baseline_x_mm,
                "baseline_mm": baseline_mm,
                "note": (
                    "pyorbbecsdk does not return LEFT_COLOR->RIGHT_COLOR extrinsics directly on this device. "
                    "This transform is the raw calibration entry reported by the device for the matched resolution."
                ),
            }
        else:
            result["calibration_reference"] = {
                "source": "device calibration table",
                "note": f"No calibration entry matched {WIDTH}x{HEIGHT}.",
            }

        output_path = os.path.join(os.path.dirname(__file__), "dual_rgb_camera_params.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nSaved to {output_path}")

        left_intrinsic = result["left_rgb"]["intrinsic"]
        right_intrinsic = result["right_rgb"]["intrinsic"]
        print("\nSummary:")
        print(
            f"Left RGB  fx={left_intrinsic['fx']:.6f}, fy={left_intrinsic['fy']:.6f}, "
            f"cx={left_intrinsic['cx']:.6f}, cy={left_intrinsic['cy']:.6f}"
        )
        print(
            f"Right RGB fx={right_intrinsic['fx']:.6f}, fy={right_intrinsic['fy']:.6f}, "
            f"cx={right_intrinsic['cx']:.6f}, cy={right_intrinsic['cy']:.6f}"
        )
        if calibration is not None:
            tx, ty, tz = calibration.transform.transform.tolist()
            print(f"Calibration transform translation (mm): [{tx:.6f}, {ty:.6f}, {tz:.6f}]")
            print(f"Baseline X (mm): {abs(tx):.6f}")
            print(f"Baseline 3D (mm): {math.sqrt(tx * tx + ty * ty + tz * tz):.6f}")

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if mode_switched and original_mode is not None:
            try:
                print(f"Restoring depth work mode: {device.get_depth_work_mode()} -> {original_mode}")
                device.set_depth_work_mode(original_mode)
            except Exception as exc:
                print(f"Failed to restore depth work mode: {exc}")


if __name__ == "__main__":
    main()
