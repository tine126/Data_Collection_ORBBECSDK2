"""Gemini 335L - List all supported camera profiles"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pyorbbecsdk import *

pipeline = Pipeline()
device = pipeline.get_device()
info = device.get_device_info()
print(f"Device: {info.get_name()}\n")

sensor_list = device.get_sensor_list()

for i in range(sensor_list.get_count()):
    sensor = sensor_list.get_sensor_by_index(i)
    sensor_type = sensor.get_type()
    print(f"{'='*60}")
    print(f"Sensor: {sensor_type}")
    print(f"{'='*60}")

    profile_list = sensor.get_stream_profile_list()
    for j in range(profile_list.get_count()):
        profile = profile_list.get_stream_profile_by_index(j)
        if hasattr(profile, 'get_width'):
            print(f"  [{j}] {profile.get_width()}x{profile.get_height()} @ {profile.get_fps()}fps | {profile.get_format()}")
    print()

print("Use these values in config.yaml")
