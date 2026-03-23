from pyorbbecsdk import *

pipeline = Pipeline()

print("=== COLOR ===")
pl = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
color_set = set()
for i in range(pl.get_count()):
    p = pl.get_stream_profile_by_index(i)
    key = (p.get_width(), p.get_height(), p.get_fps())
    if key not in color_set:
        color_set.add(key)
        print(f"  {p.get_width()}x{p.get_height()} @ {p.get_fps()} fps")

print()
print("=== DEPTH ===")
pl = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_set = set()
for i in range(pl.get_count()):
    p = pl.get_stream_profile_by_index(i)
    key = (p.get_width(), p.get_height(), p.get_fps())
    if key not in depth_set:
        depth_set.add(key)
        print(f"  {p.get_width()}x{p.get_height()} @ {p.get_fps()} fps")

print()
print("=== COMMON (same WxH @ FPS) ===")
common = color_set & depth_set
if common:
    for w, h, fps in sorted(common, key=lambda x: (x[0]*x[1], x[2])):
        print(f"  {w}x{h} @ {fps} fps")
else:
    print("  No exact match found.")

del pipeline
