"""
List all available profiles for Gemini305 camera
"""
from pyorbbecsdk import *

def list_sensor_profiles(pipeline, sensor_type, sensor_name):
    try:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        count = profile_list.get_count()
        print(f"\n{'='*60}")
        print(f"{sensor_name} - Total profiles: {count}")
        print(f"{'='*60}")

        profiles = []
        for i in range(count):
            p = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
            w, h, fps, fmt = p.get_width(), p.get_height(), p.get_fps(), p.get_format()
            profiles.append((w, h, fps, fmt))

        # Group by resolution
        from collections import defaultdict
        res_groups = defaultdict(list)
        for w, h, fps, fmt in profiles:
            res_groups[(w, h)].append((fps, fmt))

        for (w, h), fps_fmts in sorted(res_groups.items(), key=lambda x: (-x[0][0], -x[0][1])):
            print(f"\n{w}x{h}:")
            for fps, fmt in sorted(set(fps_fmts), key=lambda x: -x[0]):
                print(f"  {fps:3d} fps - {fmt}")

    except Exception as e:
        print(f"\n{sensor_name}: Not available ({e})")

def main():
    pipeline = Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()

    print(f"Device: {info.get_name()}")
    print(f"Serial: {info.get_serial_number()}")
    print(f"USB: {info.get_connection_type()}")

    list_sensor_profiles(pipeline, OBSensorType.COLOR_SENSOR, "COLOR")
    list_sensor_profiles(pipeline, OBSensorType.DEPTH_SENSOR, "DEPTH")
    list_sensor_profiles(pipeline, OBSensorType.LEFT_IR_SENSOR, "LEFT_IR")
    list_sensor_profiles(pipeline, OBSensorType.RIGHT_IR_SENSOR, "RIGHT_IR")

if __name__ == "__main__":
    main()
