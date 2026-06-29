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
    return None


def process_depth(depth_frame):
    if depth_frame is None or depth_frame.get_format() != OBFormat.Y16:
        return None, None

    w, h = depth_frame.get_width(), depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
    depth = depth.astype(np.float32) * scale
    depth = np.where((depth > 20) & (depth < 10000), depth, 0).astype(np.uint16)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth, depth_vis


def main():
    print("Starting Gemini305 camera...")
    config = Config()
    pipeline = Pipeline()

    # Gemini305: Color 1280x720 @ 10fps MJPG + Depth 1280x720 @ 5fps
    print("Configuring streams: Color 1280x720@10fps + Depth 1280x720@5fps")
    config.enable_video_stream(OBSensorType.COLOR_SENSOR, 1280, 720, 10, OBFormat.MJPG)
    config.enable_video_stream(OBSensorType.DEPTH_SENSOR, 1280, 720, 5, OBFormat.Y16)

    print("Starting pipeline...")
    pipeline.start(config)
    print("Camera started successfully!")

    print("\nPress SPACE to save frame, Q to quit")

    frame_count = 0
    output_dir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(output_dir, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames(100)

            color_img = None
            depth_raw = None
            depth_vis = None

            if frames:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame:
                    color_img = frame_to_bgr(color_frame)
                if depth_frame:
                    depth_raw, depth_vis = process_depth(depth_frame)

            if color_img is not None:
                display = color_img.copy()
                if depth_vis is not None:
                    depth_resized = cv2.resize(depth_vis, (320, 180))
                    display[0:180, 0:320] = depth_resized

                cv2.putText(display, f"Saved: {frame_count} | 1280x720", (10, display.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Gemini305 - Press SPACE to save", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and color_img is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                color_path = os.path.join(output_dir, f"{timestamp}_color.jpg")
                cv2.imwrite(color_path, color_img)

                if depth_raw is not None:
                    depth_path = os.path.join(output_dir, f"{timestamp}_depth.png")
                    cv2.imwrite(depth_path, depth_raw)

                    depth_vis_path = os.path.join(output_dir, f"{timestamp}_depth_vis.jpg")
                    cv2.imwrite(depth_vis_path, depth_vis)

                frame_count += 1
                print(f"Saved frame {frame_count}: {timestamp}")

            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()
