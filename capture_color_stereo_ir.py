import os
from datetime import datetime

import cv2
import numpy as np
from pyorbbecsdk import *


def frame_to_image(frame):
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
    if fmt == OBFormat.Y8:
        gray = np.resize(data, (height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return None


def main():
    print("Starting Gemini305 COLOR + stereo IR capture...")
    config = Config()
    pipeline = Pipeline()

    config.enable_video_stream(OBSensorType.COLOR_SENSOR, 1280, 800, 30, OBFormat.BGR)
    config.enable_video_stream(OBSensorType.LEFT_IR_SENSOR, 1280, 800, 30, OBFormat.Y8)
    config.enable_video_stream(OBSensorType.RIGHT_IR_SENSOR, 1280, 800, 30, OBFormat.Y8)

    print("Starting pipeline...")
    pipeline.start(config)
    print("Capture started.")
    print("Press SPACE to save, Q to quit")

    frame_count = 0
    output_dir = os.path.join(os.path.dirname(__file__), "color_stereo_ir")
    os.makedirs(output_dir, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            color_img = None
            left_ir = None
            right_ir = None

            if frames:
                color_frame = frames.get_color_frame()
                left_frame = frames.get_ir_frame(OBFrameType.LEFT_IR_FRAME)
                right_frame = frames.get_ir_frame(OBFrameType.RIGHT_IR_FRAME)

                if color_frame:
                    color_img = frame_to_image(color_frame)
                if left_frame:
                    left_ir = frame_to_image(left_frame)
                if right_frame:
                    right_ir = frame_to_image(right_frame)

            if color_img is not None and left_ir is not None and right_ir is not None:
                top_row = np.hstack([left_ir, color_img, right_ir])
                display = cv2.resize(top_row, (1440, 480))
                cv2.putText(
                    display,
                    f"Left IR | RGB | Right IR | Saved: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Gemini305 COLOR + Stereo IR", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and color_img is not None and left_ir is not None and right_ir is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                cv2.imwrite(os.path.join(output_dir, f"{timestamp}_color.png"), color_img)
                cv2.imwrite(os.path.join(output_dir, f"{timestamp}_left_ir.png"), left_ir)
                cv2.imwrite(os.path.join(output_dir, f"{timestamp}_right_ir.png"), right_ir)
                frame_count += 1
                print(f"Saved frame {frame_count}: {timestamp}")

            if key == ord("q") or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()
