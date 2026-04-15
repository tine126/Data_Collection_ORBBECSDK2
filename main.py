import os
import sys
import csv
import time
import json
import argparse
import threading
from datetime import datetime
from pathlib import Path

import cv2
import yaml
import numpy as np
from pyorbbecsdk import *

# ──────────────────────────── Load Config ────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Orbbec Multi-Stream Capture")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to YAML config file (default: config.yaml next to this script)",
    )
    return parser.parse_args()


# ──────────────────────────── Enum Lookups ────────────────────────────

SAMPLE_RATE_MAP = {
    "SAMPLE_RATE_1_5625_HZ": OBGyroSampleRate.SAMPLE_RATE_1_5625_HZ,
    "SAMPLE_RATE_3_125_HZ":  OBGyroSampleRate.SAMPLE_RATE_3_125_HZ,
    "SAMPLE_RATE_6_25_HZ":   OBGyroSampleRate.SAMPLE_RATE_6_25_HZ,
    "SAMPLE_RATE_12_5_HZ":   OBGyroSampleRate.SAMPLE_RATE_12_5_HZ,
    "SAMPLE_RATE_25_HZ":     OBGyroSampleRate.SAMPLE_RATE_25_HZ,
    "SAMPLE_RATE_50_HZ":     OBGyroSampleRate.SAMPLE_RATE_50_HZ,
    "SAMPLE_RATE_100_HZ":    OBGyroSampleRate.SAMPLE_RATE_100_HZ,
    "SAMPLE_RATE_200_HZ":    OBGyroSampleRate.SAMPLE_RATE_200_HZ,
    "SAMPLE_RATE_500_HZ":    OBGyroSampleRate.SAMPLE_RATE_500_HZ,
    "SAMPLE_RATE_1_KHZ":     OBGyroSampleRate.SAMPLE_RATE_1_KHZ,
    "SAMPLE_RATE_2_KHZ":     OBGyroSampleRate.SAMPLE_RATE_2_KHZ,
    "SAMPLE_RATE_4_KHZ":     OBGyroSampleRate.SAMPLE_RATE_4_KHZ,
    "SAMPLE_RATE_8_KHZ":     OBGyroSampleRate.SAMPLE_RATE_8_KHZ,
    "SAMPLE_RATE_16_KHZ":    OBGyroSampleRate.SAMPLE_RATE_16_KHZ,
    "SAMPLE_RATE_32_KHZ":    OBGyroSampleRate.SAMPLE_RATE_32_KHZ,
}

ACCEL_FS_MAP = {
    "ACCEL_FS_2g":  OBAccelFullScaleRange.ACCEL_FS_2g,
    "ACCEL_FS_4g":  OBAccelFullScaleRange.ACCEL_FS_4g,
    "ACCEL_FS_8g":  OBAccelFullScaleRange.ACCEL_FS_8g,
    "ACCEL_FS_16g": OBAccelFullScaleRange.ACCEL_FS_16g,
}

GYRO_FS_MAP = {
    "FS_16dps":   OBGyroFullScaleRange.FS_16dps,
    "FS_31dps":   OBGyroFullScaleRange.FS_31dps,
    "FS_62dps":   OBGyroFullScaleRange.FS_62dps,
    "FS_125dps":  OBGyroFullScaleRange.FS_125dps,
    "FS_250dps":  OBGyroFullScaleRange.FS_250dps,
    "FS_500dps":  OBGyroFullScaleRange.FS_500dps,
    "FS_1000dps": OBGyroFullScaleRange.FS_1000dps,
    "FS_2000dps": OBGyroFullScaleRange.FS_2000dps,
}

COLORMAP_MAP = {
    "JET":     cv2.COLORMAP_JET,
    "TURBO":   cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "BONE":    cv2.COLORMAP_BONE,
    "HOT":     cv2.COLORMAP_HOT,
}

AGGREGATE_MODE_MAP = {
    "FULL_FRAME_REQUIRE":  OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE,
    "COLOR_FRAME_REQUIRE": OBFrameAggregateOutputMode.COLOR_FRAME_REQUIRE,
    "ANY_SITUATION":       OBFrameAggregateOutputMode.ANY_SITUATION,
}

ALIGN_MODE_MAP = {
    "DISABLE": OBAlignMode.DISABLE,
    "ALIGN_D2C_HW_MODE": OBAlignMode.HW_MODE,
    "ALIGN_D2C_SW_MODE": OBAlignMode.SW_MODE,
}

FORMAT_MAP = {
    "MJPG": OBFormat.MJPG,
    "RGB":  OBFormat.RGB,
    "BGR":  OBFormat.BGR,
    "BGRA": OBFormat.BGRA,
    "RGBA": OBFormat.RGBA,
    "YUYV": OBFormat.YUYV,
    "UYVY": OBFormat.UYVY,
    "Y8":   OBFormat.Y8,
    "Y16":  OBFormat.Y16,
    "Y12":  OBFormat.Y12,
    "NV12": OBFormat.NV12,
    "NV21": OBFormat.NV21,
    "I420": OBFormat.I420,
}


def find_video_profile(profile_list, width, height, fps, fmt=None):
    """Find a matching video stream profile by resolution, fps, and optionally format."""
    count = profile_list.get_count()
    for i in range(count):
        p = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
        if p.get_width() == width and p.get_height() == height and p.get_fps() == fps:
            if fmt is None or p.get_format() == fmt:
                return p
    return None


# ──────────────────────────── Helpers ────────────────────────────

ESC_KEY = 27


def frame_to_bgr_image(color_frame):
    """Convert a color VideoFrame to a BGR numpy array."""
    if color_frame is None:
        return None
    color_frame = color_frame.as_video_frame()
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.RGB:
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    elif fmt == OBFormat.BGRA:
        img = np.resize(data, (height, width, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif fmt == OBFormat.RGBA:
        img = np.resize(data, (height, width, 4))
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif fmt == OBFormat.YUYV:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.UYVY:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    elif fmt == OBFormat.NV12:
        img = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    elif fmt == OBFormat.NV21:
        img = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV21)
    elif fmt == OBFormat.I420:
        img = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
    else:
        print(f"[WARN] Unsupported color format: {fmt}")
        return None


def process_ir_frame(ir_frame):
    """Convert an IR frame to a grayscale numpy uint8 image."""
    if ir_frame is None:
        return None
    ir_frame = ir_frame.as_video_frame()
    width = ir_frame.get_width()
    height = ir_frame.get_height()
    fmt = ir_frame.get_format()
    data = np.asanyarray(ir_frame.get_data())

    if fmt == OBFormat.Y8:
        return np.resize(data, (height, width))
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    else:
        ir_data = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
        return cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def process_depth_frame(depth_frame, min_depth, max_depth):
    """Return raw depth (uint16, mm) and a colorized visualization image."""
    if depth_frame is None:
        return None, None
    fmt = depth_frame.get_format()
    if fmt != OBFormat.Y16:
        return None, None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_data = depth_data.astype(np.float32) * scale
    depth_data = np.where((depth_data > min_depth) & (depth_data < max_depth), depth_data, 0).astype(np.uint16)

    depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return depth_data, depth_vis


def save_image(filepath, image, fmt, jpg_quality=95):
    """Save image in the specified format (png or jpg)."""
    if fmt == "jpg":
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    else:
        cv2.imwrite(filepath, image)


def save_depth_raw(filepath, depth_data, fmt):
    """Save raw depth data as png (16-bit) or npy."""
    if fmt == "npy":
        np.save(filepath, depth_data)
    else:
        cv2.imwrite(filepath, depth_data)


# ──────────────────────────── Output Directory Setup ────────────────────────────

def create_output_dirs(base_dir: str, cfg: dict):
    """Create timestamped output directory with sub-folders for each enabled modality."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_dir, f"capture_{timestamp}")
    streams = cfg["streams"]
    dirs = {"root": root}

    if streams.get("color"):
        dirs["color"] = os.path.join(root, "color")
    if streams.get("depth"):
        dirs["depth_raw"] = os.path.join(root, "depth_raw")
        dirs["depth_vis"] = os.path.join(root, "depth_vis")
    if streams.get("ir_left"):
        dirs["ir_left"] = os.path.join(root, "ir_left")
    if streams.get("ir_right"):
        dirs["ir_right"] = os.path.join(root, "ir_right")
    dirs["imu"] = root

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


# ──────────────────────────── IMU Pipeline (separate) ────────────────────────────

class IMUCollector:
    """Runs IMU capture on a separate pipeline/thread."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.running = False
        self.recording = False
        self.thread = None
        self.data_lock = threading.Lock()
        self.latest_accel = None
        self.latest_gyro = None
        self.imu_records = []
        self.available = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cfg = self.cfg
        frame_timeout = cfg["pipeline"]["frame_timeout_ms"]
        accel_rate = SAMPLE_RATE_MAP[cfg["accel"]["sample_rate"]]
        accel_fs = ACCEL_FS_MAP[cfg["accel"]["full_scale_range"]]
        gyro_rate = SAMPLE_RATE_MAP[cfg["gyro"]["sample_rate"]]
        gyro_fs = GYRO_FS_MAP[cfg["gyro"]["full_scale_range"]]
        agg_mode = AGGREGATE_MODE_MAP[cfg["pipeline"]["frame_aggregate_mode"]]

        try:
            config = Config()
            pipeline = Pipeline()
            device = pipeline.get_device()
            try:
                device.get_sensor(OBSensorType.ACCEL_SENSOR)
                device.get_sensor(OBSensorType.GYRO_SENSOR)
            except Exception:
                print("[IMU] Device does not support Accel/Gyro sensors. IMU capture disabled.")
                self.available = False
                return

            self.available = True
            config.enable_accel_stream(full_scale_range=accel_fs, sample_rate=accel_rate)
            config.enable_gyro_stream(full_scale_range=gyro_fs, sample_rate=gyro_rate)
            # IMU pipeline should not wait for all frames to sync — use ANY_SITUATION
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)
            pipeline.start(config)
            print(f"[IMU] Accel: {accel_rate}, range {accel_fs}")
            print(f"[IMU] Gyro:  {gyro_rate}, range {gyro_fs}")
            print("[IMU] IMU pipeline started.")

            while self.running:
                try:
                    # Use shorter timeout (10ms) for IMU to achieve higher sampling rate
                    frames = pipeline.wait_for_frames(10)
                    if frames is None:
                        continue

                    accel_frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
                    gyro_frame = frames.get_frame(OBFrameType.GYRO_FRAME)
                    accel_data = None
                    gyro_data = None

                    if accel_frame is not None:
                        af = accel_frame.as_accel_frame()
                        accel_data = {"ts": af.get_timestamp(), "x": af.get_x(), "y": af.get_y(), "z": af.get_z()}

                    if gyro_frame is not None:
                        gf = gyro_frame.as_gyro_frame()
                        gyro_data = {"ts": gf.get_timestamp(), "x": gf.get_x(), "y": gf.get_y(), "z": gf.get_z()}

                    with self.data_lock:
                        self.latest_accel = accel_data
                        self.latest_gyro = gyro_data
                        if self.recording and (accel_data or gyro_data):
                            record = {"sys_ts": int(time.time() * 1000)}  # milliseconds to match image timestamps
                            if accel_data:
                                record.update({f"accel_{k}": v for k, v in accel_data.items()})
                            if gyro_data:
                                record.update({f"gyro_{k}": v for k, v in gyro_data.items()})
                            self.imu_records.append(record)

                except Exception:
                    if self.running:
                        pass
            pipeline.stop()
            print("[IMU] IMU pipeline stopped.")
        except Exception as e:
            print(f"[IMU] Failed to initialize IMU pipeline: {e}")
            self.available = False

    def get_latest(self):
        with self.data_lock:
            return self.latest_accel, self.latest_gyro

    def start_recording(self):
        with self.data_lock:
            self.recording = True
            self.imu_records = []

    def stop_recording(self):
        with self.data_lock:
            self.recording = False
            return list(self.imu_records)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    @staticmethod
    def save_csv(records, filepath):
        if not records:
            print("[IMU] No IMU data to save.")
            return
        fieldnames = [
            "sys_ts",
            "accel_ts", "accel_x", "accel_y", "accel_z",
            "gyro_ts", "gyro_x", "gyro_y", "gyro_z",
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"[IMU] Saved {len(records)} records -> {filepath}")


def save_frame_timestamps(timestamps, filepath):
    """Save frame timestamps to CSV."""
    if not timestamps:
        print("[TIMESTAMPS] No frame timestamps to save.")
        return
    fieldnames = ["frame_idx", "color_ts", "depth_ts", "ir_left_ts", "ir_right_ts"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timestamps)
    print(f"[TIMESTAMPS] Saved {len(timestamps)} frame timestamps -> {filepath}")


# ──────────────────────────── Main Capture Loop ────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"[INFO] Config loaded from: {args.config}")

    # ── Read config values ──
    streams_cfg = cfg["streams"]
    color_cfg = cfg["color"]
    depth_cfg = cfg["depth"]
    ir_cfg = cfg["ir"]
    output_cfg = cfg["output"]
    preview_cfg = cfg["preview"]
    pipeline_cfg = cfg["pipeline"]

    frame_timeout = pipeline_cfg["frame_timeout_ms"]
    min_depth = depth_cfg["min_depth_mm"]
    max_depth = depth_cfg["max_depth_mm"]
    colormap = COLORMAP_MAP.get(depth_cfg.get("colormap", "JET"), cv2.COLORMAP_JET)
    jpg_quality = output_cfg.get("jpg_quality", 95)

    # ── Resolve output base dir (relative to script location) ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = output_cfg["base_dir"]
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(script_dir, base_dir)

    # ── Setup video pipeline ──
    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    print(f"[INFO] Device: {device_info.get_name()} (SN: {device_info.get_serial_number()})")

    # Enable Color stream
    has_color = False
    if streams_cfg.get("color"):
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_fmt = FORMAT_MAP.get(color_cfg.get("format")) if color_cfg.get("format") else None
            color_profile = find_video_profile(profile_list, color_cfg["width"], color_cfg["height"], color_cfg["fps"], color_fmt)
            if color_profile is None:
                print(f"[COLOR] Requested {color_cfg['width']}x{color_cfg['height']}@{color_cfg['fps']} not found, using default.")
                color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            cp = color_profile.as_video_stream_profile()
            print(f"[COLOR] Enabled: {cp.get_width()}x{cp.get_height()} @ {cp.get_fps()} fps, format: {cp.get_format()}")
            has_color = True
        except Exception as e:
            print(f"[COLOR] Not available: {e}")

    # Enable Depth stream
    has_depth = False
    if streams_cfg.get("depth"):
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_fmt = FORMAT_MAP.get(depth_cfg.get("format")) if depth_cfg.get("format") else None
            depth_profile = find_video_profile(profile_list, depth_cfg["width"], depth_cfg["height"], depth_cfg["fps"], depth_fmt)
            if depth_profile is None:
                print(f"[DEPTH] Requested {depth_cfg['width']}x{depth_cfg['height']}@{depth_cfg['fps']} not found, using default.")
                depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            dp = depth_profile.as_video_stream_profile()
            print(f"[DEPTH] Enabled: {dp.get_width()}x{dp.get_height()} @ {dp.get_fps()} fps, format: {dp.get_format()}")
            has_depth = True
        except Exception as e:
            print(f"[DEPTH] Not available: {e}")

    # Enable IR streams
    has_dual_ir = False
    has_single_ir = False
    want_ir = streams_cfg.get("ir_left") or streams_cfg.get("ir_right")
    if want_ir:
        sensor_list = device.get_sensor_list()
        for i in range(len(sensor_list)):
            stype = sensor_list[i].get_type()
            if stype == OBSensorType.LEFT_IR_SENSOR or stype == OBSensorType.RIGHT_IR_SENSOR:
                has_dual_ir = True
                break

        if has_dual_ir:
            try:
                left_profiles = pipeline.get_stream_profile_list(OBSensorType.LEFT_IR_SENSOR)
                right_profiles = pipeline.get_stream_profile_list(OBSensorType.RIGHT_IR_SENSOR)
                ir_fmt = FORMAT_MAP.get(ir_cfg.get("format")) if ir_cfg.get("format") else None
                left_ir_profile = find_video_profile(left_profiles, ir_cfg["width"], ir_cfg["height"], ir_cfg["fps"], ir_fmt)
                if left_ir_profile is None:
                    left_ir_profile = left_profiles.get_default_video_stream_profile()
                right_ir_profile = find_video_profile(right_profiles, ir_cfg["width"], ir_cfg["height"], ir_cfg["fps"], ir_fmt)
                if right_ir_profile is None:
                    right_ir_profile = right_profiles.get_default_video_stream_profile()
                if streams_cfg.get("ir_left"):
                    config.enable_stream(left_ir_profile)
                    lp = left_ir_profile.as_video_stream_profile()
                    print(f"[IR L] Enabled: {lp.get_width()}x{lp.get_height()} @ {lp.get_fps()} fps")
                if streams_cfg.get("ir_right"):
                    config.enable_stream(right_ir_profile)
                    rp = right_ir_profile.as_video_stream_profile()
                    print(f"[IR R] Enabled: {rp.get_width()}x{rp.get_height()} @ {rp.get_fps()} fps")
            except Exception as e:
                print(f"[IR] Failed to enable dual IR: {e}")
                has_dual_ir = False
        else:
            try:
                config.enable_video_stream(OBSensorType.IR_SENSOR)
                has_single_ir = True
                print("[IR] Single IR sensor enabled.")
            except Exception as e:
                print(f"[IR] Not available: {e}")

    # ── Start IMU on separate thread ──
    imu_collector = None
    if streams_cfg.get("imu"):
        imu_collector = IMUCollector(cfg)
        imu_collector.start()
        time.sleep(0.5)

    # ── Set D2C alignment mode ──
    align_mode_str = pipeline_cfg.get("align_mode", "DISABLE")
    if align_mode_str != "DISABLE":
        align_mode = ALIGN_MODE_MAP.get(align_mode_str, OBAlignMode.DISABLE)
        config.set_align_mode(align_mode)
        print(f"[D2C] Alignment mode: {align_mode_str}")

    # ── Start video pipeline ──
    pipeline.start(config)

    # ── Apply exposure settings ──
    # Color exposure
    color_ae = color_cfg.get("auto_exposure")
    if color_ae is not None:
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, bool(color_ae))
            if not color_ae:
                if color_cfg.get("exposure") is not None:
                    device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, int(color_cfg["exposure"]))
                if color_cfg.get("gain") is not None:
                    device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, int(color_cfg["gain"]))
            ae_str = "AUTO" if color_ae else f"Manual (exposure={color_cfg.get('exposure')}, gain={color_cfg.get('gain')})"
            print(f"[COLOR] Exposure: {ae_str}")
        except Exception as e:
            print(f"[COLOR] Failed to set exposure: {e}")

    # Depth exposure
    depth_ae = depth_cfg.get("auto_exposure")
    if depth_ae is not None:
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, bool(depth_ae))
            if not depth_ae:
                if depth_cfg.get("exposure") is not None:
                    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, int(depth_cfg["exposure"]))
                if depth_cfg.get("gain") is not None:
                    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_GAIN_INT, int(depth_cfg["gain"]))
            ae_str = "AUTO" if depth_ae else f"Manual (exposure={depth_cfg.get('exposure')}, gain={depth_cfg.get('gain')})"
            print(f"[DEPTH] Exposure: {ae_str}")
        except Exception as e:
            print(f"[DEPTH] Failed to set exposure: {e}")

    # IR exposure
    ir_ae = ir_cfg.get("auto_exposure")
    if ir_ae is not None:
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_IR_AUTO_EXPOSURE_BOOL, bool(ir_ae))
            if not ir_ae:
                if ir_cfg.get("exposure") is not None:
                    device.set_int_property(OBPropertyID.OB_PROP_IR_EXPOSURE_INT, int(ir_cfg["exposure"]))
                if ir_cfg.get("gain") is not None:
                    device.set_int_property(OBPropertyID.OB_PROP_IR_GAIN_INT, int(ir_cfg["gain"]))
            ae_str = "AUTO" if ir_ae else f"Manual (exposure={ir_cfg.get('exposure')}, gain={ir_cfg.get('gain')})"
            print(f"[IR] Exposure: {ae_str}")
        except Exception as e:
            print(f"[IR] Failed to set exposure: {e}")

    # ── Laser control: turn off laser if IR streams are enabled ──
    if want_ir:
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, False)
            print("[LASER] Laser turned OFF for IR capture.")
        except Exception as e:
            print(f"[LASER] OB_PROP_LASER_BOOL failed: {e}")
            try:
                device.set_bool_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 0)
                print("[LASER] Laser turned OFF (via LASER_CONTROL_INT).")
            except Exception as e2:
                print(f"[LASER] Could not disable laser: {e2}")

    print("[INFO] Video pipeline started.")
    print()
    print("=" * 60)
    print("  Controls:")
    print("    SPACE  - Start / Stop recording")
    print("    U      - Record 200 frames then auto-stop")
    print("    S      - Save single snapshot")
    print("    Q/ESC  - Quit")
    print("=" * 60)
    print()

    # ── State ──
    recording = False
    auto_stop_at = None  # Auto-stop when frame_idx reaches this value
    frame_idx = 0
    output_dirs = None
    record_start_time = None

    # ── Window setup ──
    preview_enabled = preview_cfg.get("enabled", True)
    panel_w = preview_cfg.get("panel_width", 480)
    panel_h = preview_cfg.get("panel_height", 360)
    if preview_enabled:
        win_name = "Orbbec Multi-Stream Capture"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, preview_cfg["window_width"], preview_cfg["window_height"])
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)  # Keep window on top initially

    # ── Show initial black frame so window isn't gray ──
    if preview_enabled:
        init_mosaic = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
        cv2.putText(init_mosaic, "Waiting for camera...", (panel_w - 150, panel_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        cv2.imshow(win_name, init_mosaic)
        cv2.waitKey(1)

    try:
        while True:
            frames = pipeline.wait_for_frames(frame_timeout)

            # ── Grab frames (if available) ──
            color_image = None
            depth_raw = None
            depth_vis = None
            ir_left_image = None
            ir_right_image = None
            color_ts = None
            depth_ts = None
            ir_left_ts = None
            ir_right_ts = None

            if frames is not None:
                # Get system timestamp for this frame
                sys_ts = int(time.time() * 1000)  # milliseconds

                if has_color:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_ts = color_frame.get_timestamp() or sys_ts
                        color_image = frame_to_bgr_image(color_frame)

                if has_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_ts = depth_frame.get_timestamp() or sys_ts
                        depth_raw, depth_vis_raw = process_depth_frame(depth_frame, min_depth, max_depth)
                        if depth_vis_raw is not None:
                            depth_vis = cv2.applyColorMap(depth_vis_raw, colormap)

                if has_dual_ir:
                    if streams_cfg.get("ir_left"):
                        left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
                        if left_frame:
                            ir_left_ts = left_frame.get_timestamp() or sys_ts
                            ir_left_image = process_ir_frame(left_frame)
                    if streams_cfg.get("ir_right"):
                        right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
                        if right_frame:
                            ir_right_ts = right_frame.get_timestamp() or sys_ts
                            ir_right_image = process_ir_frame(right_frame)
                elif has_single_ir:
                    ir_frame = frames.get_frame(OBFrameType.IR_FRAME)
                    if ir_frame:
                        ir_left_ts = ir_frame.get_timestamp() or sys_ts
                        ir_left_image = process_ir_frame(ir_frame)

            # ── Save data if recording ──
            if recording and output_dirs and frames is not None:
                # Use device timestamp as filename (use depth timestamp as primary)
                # Select timestamp with priority: depth > color > ir_left
                device_ts = depth_ts or color_ts or ir_left_ts or ir_right_ts
                ts_str = str(int(device_ts)) if device_ts else f"{frame_idx:06d}"

                color_fmt = output_cfg["color_format"]
                depth_raw_fmt = output_cfg["depth_raw_format"]
                depth_vis_fmt = output_cfg["depth_vis_format"]
                ir_fmt = output_cfg["ir_format"]

                if color_image is not None:
                    save_image(os.path.join(output_dirs["color"], f"{ts_str}.{color_fmt}"), color_image, color_fmt, jpg_quality)

                if depth_raw is not None:
                    ext = "npy" if depth_raw_fmt == "npy" else depth_raw_fmt
                    save_depth_raw(os.path.join(output_dirs["depth_raw"], f"{ts_str}.{ext}"), depth_raw, depth_raw_fmt)
                if depth_vis is not None:
                    save_image(os.path.join(output_dirs["depth_vis"], f"{ts_str}.{depth_vis_fmt}"), depth_vis, depth_vis_fmt, jpg_quality)

                if ir_left_image is not None and "ir_left" in output_dirs:
                    save_image(os.path.join(output_dirs["ir_left"], f"{ts_str}.{ir_fmt}"), ir_left_image, ir_fmt, jpg_quality)
                if ir_right_image is not None and "ir_right" in output_dirs:
                    save_image(os.path.join(output_dirs["ir_right"], f"{ts_str}.{ir_fmt}"), ir_right_image, ir_fmt, jpg_quality)

                frame_idx += 1

            # ── Auto-stop check ──
            if recording and auto_stop_at is not None and frame_idx >= auto_stop_at:
                recording = False
                auto_stop_at = None
                imu_records = imu_collector.stop_recording() if imu_collector else []
                if output_dirs:
                    if imu_records:
                        IMUCollector.save_csv(imu_records, os.path.join(output_dirs["imu"], "imu_data.csv"))
                    meta = {
                        "device": device_info.get_name(),
                        "serial_number": device_info.get_serial_number(),
                        "config_file": args.config,
                        "total_frames": frame_idx,
                        "duration_sec": time.time() - record_start_time,
                        "streams": {
                            "color": has_color,
                            "depth": has_depth,
                            "ir_dual": has_dual_ir,
                            "ir_single": has_single_ir,
                            "imu": imu_collector.available if imu_collector else False,
                        },
                        "imu_records_count": len(imu_records),
                    }
                    with open(os.path.join(output_dirs["root"], "metadata.json"), "w") as f:
                        json.dump(meta, f, indent=2)
                    print(f"[AUTO-STOP] Reached {frame_idx} frames -> {output_dirs['root']}")

            # ── Build preview mosaic ──
            if preview_enabled:
                panels = []

                if color_image is not None:
                    p = cv2.resize(color_image, (panel_w, panel_h))
                    cv2.putText(p, "COLOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    panels.append(p)

                if depth_vis is not None:
                    p = cv2.resize(depth_vis, (panel_w, panel_h))
                    cv2.putText(p, "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    panels.append(p)

                if ir_left_image is not None:
                    p = cv2.resize(cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR), (panel_w, panel_h))
                    label = "IR LEFT" if has_dual_ir else "IR"
                    cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    panels.append(p)

                if ir_right_image is not None:
                    p = cv2.resize(cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR), (panel_w, panel_h))
                    cv2.putText(p, "IR RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    panels.append(p)

                if panels:
                    while len(panels) < 4:
                        panels.append(np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
                    row1 = np.hstack(panels[:2])
                    row2 = np.hstack(panels[2:4])
                    mosaic = np.vstack([row1, row2])

                    if recording:
                        elapsed = time.time() - record_start_time
                        rec_text = f"REC  Frame: {frame_idx}  Time: {elapsed:.1f}s"
                        cv2.putText(mosaic, rec_text, (10, mosaic.shape[0] - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.circle(mosaic, (mosaic.shape[1] - 30, 30), 12, (0, 0, 255), -1)

                    if imu_collector:
                        accel, gyro = imu_collector.get_latest()
                        if accel:
                            imu_text = f"Accel: x={accel['x']:.3f} y={accel['y']:.3f} z={accel['z']:.3f}"
                            cv2.putText(mosaic, imu_text, (10, mosaic.shape[0] - 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        if gyro:
                            imu_text = f"Gyro:  x={gyro['x']:.3f} y={gyro['y']:.3f} z={gyro['z']:.3f}"
                            cv2.putText(mosaic, imu_text, (10, mosaic.shape[0] - 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    cv2.imshow(win_name, mosaic)

            # ── Keyboard handling ──
            if preview_enabled:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF
                time.sleep(0.001)

            if key == ord('q') or key == ord('Q') or key == ESC_KEY:
                break

            if (key == ord('u') or key == ord('U')) and not recording:
                # U key: Record 200 frames then auto-stop
                output_dirs = create_output_dirs(base_dir, cfg)
                frame_idx = 0
                record_start_time = time.time()
                recording = True
                auto_stop_at = 200
                if imu_collector:
                    imu_collector.start_recording()
                print(f"[REC] Auto-recording 200 frames -> {output_dirs['root']}")

            if key == ord(' '):
                if not recording:
                    output_dirs = create_output_dirs(base_dir, cfg)
                    frame_idx = 0
                    record_start_time = time.time()
                    recording = True
                    if imu_collector:
                        imu_collector.start_recording()
                    print(f"[REC] Recording started -> {output_dirs['root']}")
                else:
                    recording = False
                    imu_records = imu_collector.stop_recording() if imu_collector else []
                    if output_dirs:
                        if imu_records:
                            IMUCollector.save_csv(imu_records, os.path.join(output_dirs["imu"], "imu_data.csv"))
                        meta = {
                            "device": device_info.get_name(),
                            "serial_number": device_info.get_serial_number(),
                            "config_file": args.config,
                            "total_frames": frame_idx,
                            "duration_sec": time.time() - record_start_time,
                            "streams": {
                                "color": has_color,
                                "depth": has_depth,
                                "ir_dual": has_dual_ir,
                                "ir_single": has_single_ir,
                                "imu": imu_collector.available if imu_collector else False,
                            },
                            "imu_records_count": len(imu_records),
                        }
                        with open(os.path.join(output_dirs["root"], "metadata.json"), "w") as f:
                            json.dump(meta, f, indent=2)
                        print(f"[REC] Stopped. {frame_idx} frames saved -> {output_dirs['root']}")

            if (key == ord('s') or key == ord('S')) and not recording:
                snap_dirs = create_output_dirs(base_dir, cfg)
                snap_ts = "snapshot"
                color_fmt = output_cfg["color_format"]
                depth_raw_fmt = output_cfg["depth_raw_format"]
                depth_vis_fmt = output_cfg["depth_vis_format"]
                ir_fmt = output_cfg["ir_format"]

                if color_image is not None:
                    save_image(os.path.join(snap_dirs["color"], f"{snap_ts}.{color_fmt}"), color_image, color_fmt, jpg_quality)
                if depth_raw is not None:
                    ext = "npy" if depth_raw_fmt == "npy" else depth_raw_fmt
                    save_depth_raw(os.path.join(snap_dirs["depth_raw"], f"{snap_ts}.{ext}"), depth_raw, depth_raw_fmt)
                if depth_vis is not None:
                    save_image(os.path.join(snap_dirs["depth_vis"], f"{snap_ts}.{depth_vis_fmt}"), depth_vis, depth_vis_fmt, jpg_quality)
                if ir_left_image is not None and "ir_left" in snap_dirs:
                    save_image(os.path.join(snap_dirs["ir_left"], f"{snap_ts}.{ir_fmt}"), ir_left_image, ir_fmt, jpg_quality)
                if ir_right_image is not None and "ir_right" in snap_dirs:
                    save_image(os.path.join(snap_dirs["ir_right"], f"{snap_ts}.{ir_fmt}"), ir_right_image, ir_fmt, jpg_quality)
                if imu_collector:
                    accel, gyro = imu_collector.get_latest()
                    if accel or gyro:
                        imu_snap = {"accel": accel, "gyro": gyro, "sys_ts": time.time()}
                        with open(os.path.join(snap_dirs["root"], "imu_snapshot.json"), "w") as f:
                            json.dump(imu_snap, f, indent=2)
                print(f"[SNAP] Snapshot saved -> {snap_dirs['root']}")

    except KeyboardInterrupt:
        pass
    finally:
        if recording:
            imu_records = imu_collector.stop_recording() if imu_collector else []
            if output_dirs and imu_records:
                IMUCollector.save_csv(imu_records, os.path.join(output_dirs["imu"], "imu_data.csv"))
            print(f"[REC] Emergency stop. {frame_idx} frames saved.")

        if imu_collector:
            imu_collector.stop()
        # Restore laser before stopping pipeline
        if want_ir:
            try:
                device.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, True)
                print("[LASER] Laser restored ON.")
            except Exception:
                try:
                    device.set_bool_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 1)
                    print("[LASER] Laser restored ON (via LASER_CONTROL_INT).")
                except Exception:
                    pass
        pipeline.stop()
        if preview_enabled:
            cv2.destroyAllWindows()
        print("[INFO] All pipelines stopped. Bye!")


if __name__ == "__main__":
    main()
