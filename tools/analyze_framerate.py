import os
import json

capture_dir = r"F:\Code\orbbecSDKv2\output\capture_20260327_101930"

# Get color image timestamps
color_dir = os.path.join(capture_dir, "color")
files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])
timestamps = [int(f.split('.')[0]) for f in files]

if len(timestamps) > 1:
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    avg_interval = sum(intervals) / len(intervals)
    fps = 1000 / avg_interval  # timestamps in ms

    duration = (timestamps[-1] - timestamps[0]) / 1000  # seconds

    print(f"总帧数: {len(timestamps)}")
    print(f"采集时长: {duration:.2f} 秒")
    print(f"平均帧间隔: {avg_interval:.2f} ms")
    print(f"实际帧率: {fps:.2f} fps")
    print(f"目标帧率: 30 fps")
    print(f"达成率: {fps/30*100:.1f}%")

# Check IMU rate
with open(os.path.join(capture_dir, "metadata.json")) as f:
    meta = json.load(f)
    imu_count = meta.get("imu_records_count", 0)
    if imu_count > 0:
        imu_rate = imu_count / duration
        print(f"\nIMU采样数: {imu_count}")
        print(f"IMU采样率: {imu_rate:.2f} Hz")
        print(f"目标采样率: 200 Hz")
