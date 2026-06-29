import os
from datetime import datetime
import cv2
import numpy as np
from pyorbbecsdk import *


def frame_to_bgr(frame):
    if frame is None:
        return None
    frame = frame.as_video_frame()
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
        return cv2.cvtColor(np.resize(data, (h, w, 4)), cv2.COLOR_BGRA2BGR)
    return None


def main():
    print("Testing Gemini 335L dual RGB...")
    pipeline = Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()
    print(f"Device: {info.get_name()}")

    # 尝试方法1: 使用LEFT_COLOR_STREAM和RIGHT_COLOR_STREAM
    print("\n尝试启用LEFT_COLOR_STREAM和RIGHT_COLOR_STREAM...")
    config = Config()

    try:
        config.enable_video_stream(OBStreamType.LEFT_COLOR_STREAM, 1280, 720, 30, OBFormat.BGR)
        config.enable_video_stream(OBStreamType.RIGHT_COLOR_STREAM, 1280, 720, 30, OBFormat.BGR)

        print("配置成功，启动pipeline...")
        pipeline.start(config)
        print("启动成功！双RGB模式已激活")

        output_dir = os.path.join(os.path.dirname(__file__), "dual_rgb_335L")
        os.makedirs(output_dir, exist_ok=True)
        frame_count = 0
        capturing = False
        session_timestamp = None

        print("Press SPACE to start/stop capture, Q to quit")

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
                    cv2.imwrite(os.path.join(output_dir, f"{session_timestamp}_{frame_count:06d}_left_rgb.png"), left_img)
                    cv2.imwrite(os.path.join(output_dir, f"{session_timestamp}_{frame_count:06d}_right_rgb.png"), right_img)
                    frame_count += 1

                display = np.hstack([left_img, right_img])
                status = "CAPTURING" if capturing else "READY"
                color = (0, 0, 255) if capturing else (0, 255, 0)
                cv2.putText(display, f"{status} | Frames: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Gemini 335L Dual RGB", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not capturing:
                    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    capturing = True
                    frame_count = 0
                    print(f"Started capturing: {session_timestamp}")
                else:
                    capturing = False
                    print(f"Stopped: {frame_count} frames saved")

            if key == ord("q") or key == 27:
                break

        pipeline.stop()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"失败: {e}")
        print("\nGemini 335L可能不支持双RGB模式")
        print("硬件配置: 1个COLOR传感器 + 2个IR传感器(灰度)")


if __name__ == "__main__":
    main()
