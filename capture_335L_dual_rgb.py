import os
from datetime import datetime
import cv2
import numpy as np
from pyorbbecsdk import *


def frame_to_bgr(frame):
    if frame is None:
        return None
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.RGB:
        return cv2.cvtColor(np.resize(data, (h, w, 3)), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return np.resize(data, (h, w, 3))
    elif fmt == OBFormat.BGRA:
        bgra = np.resize(data, (h, w, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return None


def main():
    print("检测Gemini 335L双RGB支持...")
    pipeline = Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()
    print(f"设备: {info.get_name()}")

    # 检测是否支持双RGB
    sensor_list = device.get_sensor_list()
    support_dual_rgb = False
    for i in range(sensor_list.get_count()):
        sensor_type = sensor_list.get_sensor_by_index(i).get_type()
        if sensor_type in [OBSensorType.LEFT_COLOR_SENSOR, OBSensorType.RIGHT_COLOR_SENSOR]:
            support_dual_rgb = True
            break

    if not support_dual_rgb:
        print("错误: 设备不支持双RGB模式")
        print("可用传感器:", [sensor_list.get_sensor_by_index(i).get_type()
                           for i in range(sensor_list.get_count())])
        return

    print("✓ 支持双RGB模式")

    # 配置双RGB流
    config = Config()
    try:
        config.enable_stream(OBSensorType.LEFT_COLOR_SENSOR)
        config.enable_stream(OBSensorType.RIGHT_COLOR_SENSOR)
    except Exception as e:
        print(f"配置失败: {e}")
        return

    print("启动pipeline...")
    pipeline.start(config)
    print("双RGB已启动！")
    print("按空格开始/停止连续采集，Q退出")

    output_dir = os.path.join(os.path.dirname(__file__), "dual_rgb_335L")
    os.makedirs(output_dir, exist_ok=True)

    capturing = False
    frame_count = 0
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
                    left_img = frame_to_bgr(left_frame.as_video_frame())
                if right_frame:
                    right_img = frame_to_bgr(right_frame.as_video_frame())

            if left_img is not None and right_img is not None:
                if capturing:
                    left_path = os.path.join(output_dir, f"{session_timestamp}_{frame_count:06d}_left.png")
                    right_path = os.path.join(output_dir, f"{session_timestamp}_{frame_count:06d}_right.png")
                    cv2.imwrite(left_path, left_img)
                    cv2.imwrite(right_path, right_img)
                    frame_count += 1

                display = np.hstack([left_img, right_img])
                status = "CAPTURING" if capturing else "READY"
                color = (0, 0, 255) if capturing else (0, 255, 0)
                cv2.putText(display, f"{status} | Frames: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Gemini 335L Dual RGB", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not capturing:
                    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frame_count = 0
                    capturing = True
                    print(f"开始采集: {session_timestamp}")
                else:
                    capturing = False
                    print(f"停止采集: 已保存{frame_count}帧")

            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"总计保存: {frame_count}帧")


if __name__ == "__main__":
    main()
