"""Query current exposure/gain parameters from the Orbbec camera."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyorbbecsdk import *


def main():
    pipeline = Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()
    print(f"Device: {info.get_name()} (SN: {info.get_serial_number()})")
    print("=" * 60)

    # Properties to query: (property_id, label, getter_type)
    bool_props = [
        (OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "Color Auto Exposure"),
        (OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, "Depth Auto Exposure"),
        (OBPropertyID.OB_PROP_IR_AUTO_EXPOSURE_BOOL,    "IR Auto Exposure"),
    ]

    int_props = [
        (OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,  "Color Exposure"),
        (OBPropertyID.OB_PROP_COLOR_GAIN_INT,      "Color Gain"),
        (OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,  "Depth Exposure"),
        (OBPropertyID.OB_PROP_DEPTH_GAIN_INT,      "Depth Gain"),
        (OBPropertyID.OB_PROP_IR_EXPOSURE_INT,     "IR Exposure"),
        (OBPropertyID.OB_PROP_IR_GAIN_INT,         "IR Gain"),
    ]

    print("\n-- Auto Exposure Switches --")
    for prop_id, label in bool_props:
        try:
            val = device.get_bool_property(prop_id)
            print(f"  {label}: {val}")
        except Exception as e:
            print(f"  {label}: N/A ({e})")

    print("\n-- Exposure & Gain Values --")
    for prop_id, label in int_props:
        try:
            val = device.get_int_property(prop_id)
            print(f"  {label}: {val}")
        except Exception as e:
            print(f"  {label}: N/A ({e})")

    # Also try to get the property ranges
    print("\n-- Exposure & Gain Ranges --")
    for prop_id, label in int_props:
        try:
            r = device.get_int_property_range(prop_id)
            print(f"  {label}: min={r.min}, max={r.max}, step={r.step}, default={r.default_value}")
        except Exception as e:
            print(f"  {label} range: N/A ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
