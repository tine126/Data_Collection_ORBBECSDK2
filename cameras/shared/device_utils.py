"""Device detection and capability utilities."""
from pyorbbecsdk import OBSensorType


def get_sensor_types(device):
    """Get all sensor types available on device."""
    sensor_list = device.get_sensor_list()
    return [sensor_list.get_sensor_by_index(i).get_type()
            for i in range(sensor_list.get_count())]


def has_dual_color_sensors(device):
    """Check if device supports dual color sensors (335L)."""
    sensor_types = get_sensor_types(device)
    return (hasattr(OBSensorType, "LEFT_COLOR_SENSOR") and
            hasattr(OBSensorType, "RIGHT_COLOR_SENSOR") and
            OBSensorType.LEFT_COLOR_SENSOR in sensor_types and
            OBSensorType.RIGHT_COLOR_SENSOR in sensor_types)


def find_depth_work_mode(device, target_name):
    """Find depth work mode by name."""
    mode_list = device.get_depth_work_mode_list()
    for i in range(mode_list.get_count()):
        mode = mode_list.get_depth_work_mode_by_index(i)
        if mode.get_name() == target_name:
            return mode
    return None
