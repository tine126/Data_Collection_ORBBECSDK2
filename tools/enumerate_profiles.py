"""
Enumerate all supported stream profiles for the connected Orbbec camera.
Run this to see exactly which resolutions, formats, and frame rates your device supports.

Usage:
    conda activate basketball_env
    python enumerate_profiles.py
"""

from pyorbbecsdk import *


def main():
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No Orbbec device found!")
        return

    device = device_list.get_device_by_index(0)
    info = device.get_device_info()
    print("=" * 70)
    print(f"  Device : {info.get_name()}")
    print(f"  SN     : {info.get_serial_number()}")
    print(f"  FW     : {info.get_firmware_version()}")
    print("=" * 70)

    sensor_list = device.get_sensor_list()
    for i in range(len(sensor_list)):
        sensor = sensor_list[i]
        sensor_type = sensor.get_type()
        print(f"\n--- Sensor: {sensor_type} ---")

        # IMU sensors don't have video stream profiles
        if sensor_type in (OBSensorType.ACCEL_SENSOR, OBSensorType.GYRO_SENSOR):
            print("  (IMU sensor - use enable_accel_stream / enable_gyro_stream)")
            continue

        try:
            pipeline = Pipeline(device)
            profile_list = pipeline.get_stream_profile_list(sensor_type)
            count = profile_list.get_count()
            print(f"  Supported profiles ({count}):")
            print(f"  {'#':>4}  {'Format':<12} {'Resolution':<14} {'FPS':>4}")
            print(f"  {'----':>4}  {'----------':<12} {'------------':<14} {'---':>4}")
            for j in range(count):
                profile = profile_list.get_video_stream_profile_by_index(j)
                w = profile.get_width()
                h = profile.get_height()
                fmt = profile.get_format()
                fps = profile.get_fps()
                print(f"  {j:4d}  {str(fmt):<12} {w}x{h:<10} {fps:4d}")
            del pipeline
        except Exception as e:
            print(f"  Could not enumerate: {e}")


if __name__ == "__main__":
    main()
