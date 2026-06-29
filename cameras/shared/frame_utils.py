"""Frame processing utilities shared across camera models."""
import cv2
import numpy as np
from pyorbbecsdk import OBFormat


def frame_to_bgr(frame):
    """Convert Orbbec video frame to BGR numpy array.

    Note: Expects frame to already be a video frame (call as_video_frame() before passing).
    """
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
        return cv2.cvtColor(np.resize(data, (h, w, 4)), cv2.COLOR_BGRA2BGR)
    return None


def visualize_depth(depth_data, min_depth=20, max_depth=10000, colormap=cv2.COLORMAP_JET):
    """Visualize depth data with colormap."""
    depth_clipped = np.clip(depth_data, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, colormap)
