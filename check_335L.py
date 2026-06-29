from pyorbbecsdk import *

pipeline = Pipeline()
device = pipeline.get_device()
info = device.get_device_info()

print(f"设备: {info.get_name()}")
print(f"固件: {info.get_firmware_version()}")

print("\n可用传感器:")
sensor_list = device.get_sensor_list()
for i in range(sensor_list.get_count()):
    sensor = sensor_list.get_sensor_by_index(i)
    print(f"  {i}: {sensor.get_type()}")

print("\n检查COLOR相关传感器配置:")
for sensor_type in [OBSensorType.COLOR_SENSOR, OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR]:
    try:
        profiles = pipeline.get_stream_profile_list(sensor_type)
        count = profiles.get_count()
        print(f"\n{sensor_type}: {count} 个配置")

        # 显示1280x720的配置
        print("  1280x720可用配置:")
        for i in range(count):
            p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
            if p.get_width() == 1280 and p.get_height() == 720:
                print(f"    {p.get_fps()} fps - {p.get_format()}")
    except Exception as e:
        print(f"{sensor_type}: 不可用 ({e})")
