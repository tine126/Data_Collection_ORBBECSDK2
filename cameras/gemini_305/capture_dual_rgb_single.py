import os
from datetime import datetime

import cv2
import numpy as np
from pyorbbecsdk import *


def frame_to_bgr(frame):
    if frame is None:
        return None

    frame = frame.as_video_frame()
    width, height = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(np.resize(data, (height, width, 3)), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if fmt == OBFormat.BGRA:
        return cv2.cvtColor(np.resize(data, (height, width, 4)), cv2.COLOR_BGRA2BGR)

    return None


DUAL_COLOR_MODE_NAME = "Dual Color Streams"


def get_sensor_types(device):
    sensor_list = device.get_sensor_list()
    return [sensor_list.get_sensor_by_index(i).get_type() for i in range(sensor_list.get_count())]


def has_dual_color_sensors(device):
    sensor_types = get_sensor_types(device)
    return (
        hasattr(OBSensorType, "LEFT_COLOR_SENSOR")
        and hasattr(OBSensorType, "RIGHT_COLOR_SENSOR")
        and OBSensorType.LEFT_COLOR_SENSOR in sensor_types
        and OBSensorType.RIGHT_COLOR_SENSOR in sensor_types
    )


def find_depth_work_mode(device, target_name):
    mode_list = device.get_depth_work_mode_list()
    for i in range(mode_list.get_count()):
        mode = mode_list.get_depth_work_mode_by_index(i)
        if str(mode) == target_name:
            return mode
    return None


def ensure_dual_color_mode(device):
    original_mode = device.get_depth_work_mode()
    if has_dual_color_sensors(device):
        return original_mode, False

    dual_color_mode = find_depth_work_mode(device, DUAL_COLOR_MODE_NAME)
    if dual_color_mode is None:
        info = device.get_device_info()
        raise RuntimeError(f"{info.get_name()} 不支持 {DUAL_COLOR_MODE_NAME} 模式。")

    print(f"Switching depth work mode: {original_mode} -> {dual_color_mode}")
    device.set_depth_work_mode(dual_color_mode)

    if not has_dual_color_sensors(device):
        sensor_names = [str(sensor_type) for sensor_type in get_sensor_types(device)]
        raise RuntimeError(f"切换到 {DUAL_COLOR_MODE_NAME} 后仍未发现左右 RGB 传感器: {sensor_names}")

    return original_mode, True


def main():
    print("Starting dual RGB capture...")
    pipeline = Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()
    print(f"Device: {info.get_name()}")

    original_mode, mode_switched = ensure_dual_color_mode(device)

    config = Config()
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    config.enable_video_stream(OBSensorType.LEFT_COLOR_SENSOR, 1280, 800, 30, OBFormat.BGR)
    config.enable_video_stream(OBSensorType.RIGHT_COLOR_SENSOR, 1280, 800, 30, OBFormat.BGR)

    print("Starting pipeline...")
    pipeline.start(config)
    print("Dual RGB started.")
    print("Press SPACE to start/stop continuous capture, Q to quit")

    output_dir = os.path.join(os.path.dirname(__file__), "dual_rgb")
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    capturing = False
    session_timestamp = None

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            left_img = None
            right_img = None

            if frames:
                left_frame = frames.get_frame_by_type(OBFrameType.LEFT_COLOR_FRAME)
                right_frame = frames.get_frame_by_type(OBFrameType.RIGHT_COLOR_FRAME)
                if left_frame:
                    left_img = frame_to_bgr(left_frame)
                if right_frame:
                    right_img = frame_to_bgr(right_frame)

            if left_img is not None and right_img is not None:
                if capturing:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    cv2.imwrite(os.path.join(output_dir, f"{session_timestamp}_{frame_count:06d}_left_rgb.png"), left_img)
                    cv2.imwrite(os.path.join(output_dir, f"{session_timestamp}_{frame_count:06d}_right_rgb.png"), right_img)
                    frame_count += 1

                display = np.hstack([left_img, right_img])
                status = "CAPTURING" if capturing else "READY"
                color = (0, 0, 255) if capturing else (0, 255, 0)
                cv2.putText(display, f"{status} | Frames: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Dual RGB Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not capturing:
                    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    capturing = True
                    frame_count = 0
                    print(f"Started capturing: {session_timestamp}")
                else:
                    capturing = False
                    print(f"Stopped capturing: {frame_count} frames saved")

            if key == ord("q") or key == 27:
                break
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
        cv2.destroyAllWindows()
        print(f"Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()
