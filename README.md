# Orbbec SDK v2 数据采集工具使用文档

## 环境要求

- Python 3.8+
- 依赖：`pyorbbecsdk`, `opencv-python`, `numpy`, `pyyaml`

## 安装依赖

```bash
conda activate env
pip install pyyaml opencv-python numpy
```

## 配置文件

配置文件位于 `config.yaml`，可配置以下参数：

### 数据流开关
```yaml
streams:
  color: true      # 彩色图
  depth: true      # 深度图
  ir_left: true    # 左红外
  ir_right: true   # 右红外
  imu: true        # IMU数据
```

### 分辨率和帧率
```yaml
color:
  width: 848
  height: 480
  fps: 30

depth:
  width: 848
  height: 480
  fps: 30
  min_depth_mm: 20
  max_depth_mm: 10000
  colormap: "JET"  # 可选: JET, TURBO, INFERNO, BONE, HOT

ir:
  width: 848
  height: 480
  fps: 30
```

**注意**：
- Gemini 335L 要求 Depth 和 IR 必须使用相同的分辨率和帧率
- 推荐配置：848x480@30fps（稳定性好，支持 D2C 对齐）
- 其他支持的配置可通过 `tools/find_common_profiles.py` 查看

### IMU 配置
```yaml
accel:
  sample_rate: "SAMPLE_RATE_200_HZ"
  full_scale_range: "ACCEL_FS_4g"

gyro:
  sample_rate: "SAMPLE_RATE_200_HZ"
  full_scale_range: "FS_1000dps"
```

### D2C 对齐配置
```yaml
pipeline:
  align_mode: "ALIGN_D2C_SW_MODE"  # D2C对齐模式
```

**对齐模式说明**：
- `DISABLE` - 关闭对齐（深度和彩色在各自坐标系）
- `ALIGN_D2C_HW_MODE` - 硬件对齐（快速，需要相同分辨率）
- `ALIGN_D2C_SW_MODE` - 软件对齐（兼容性好，推荐）

启用 D2C 后，深度图会对齐到彩色图坐标系，像素位置一一对应。

### 输出设置
```yaml
output:
  base_dir: "./output"
  color_format: "png"       # png 或 jpg
  depth_raw_format: "png"   # png 或 npy
  depth_vis_format: "png"
  ir_format: "png"
  jpg_quality: 95
```

## 使用方法

### 1. 数据采集 (main.py)

启动采集程序：
```bash
python main.py
```

或指定配置文件：
```bash
python main.py -c my_config.yaml
```

#### 快捷键

| 按键 | 功能 |
|------|------|
| **U** | 自动录制 200 帧后停止 |
| **SPACE** | 手动开始/停止录制 |
| **S** | 保存单帧快照 |
| **Q / ESC** | 退出程序 |

#### 输出目录结构

```
output/
└── capture_YYYYMMDD_HHMMSS/
    ├── color/           # 彩色图 (png/jpg)
    ├── depth_raw/       # 原始深度数据 (png/npy)
    ├── depth_vis/       # 深度可视化 (png/jpg)
    ├── ir_left/         # 左红外图 (png/jpg)
    ├── ir_right/        # 右红外图 (png/jpg)
    ├── imu_data.csv     # IMU数据
    └── metadata.json    # 采集元数据
```

### 2. 查看支持的分辨率 (tools/enumerate_profiles.py)

```bash
python tools/enumerate_profiles.py
```

输出设备支持的所有分辨率、格式和帧率。

### 3. 查找通用分辨率 (tools/find_common_profiles.py)

```bash
python tools/find_common_profiles.py
```

输出 Color 和 Depth 都支持的分辨率和帧率组合。

### 4. 获取相机内参 (tools/get_camera_intrinsics.py)

```bash
python tools/get_camera_intrinsics.py
```

```bash
python tools/get_camera_intrinsics.py
```

输出并保存相机内参到 `camera_intrinsics.json`：
- fx, fy: 焦距
- cx, cy: 主点坐标
- width, height: 分辨率

## 数据格式说明

### IMU 数据 (imu_data.csv)

| 字段 | 说明 |
|------|------|
| sys_ts | 系统时间戳 (秒) |
| accel_ts | 加速度计时间戳 (ms) |
| accel_x/y/z | 加速度 (m/s²) |
| gyro_ts | 陀螺仪时间戳 (ms) |
| gyro_x/y/z | 角速度 (rad/s) |

### 深度数据

- **depth_raw**: 原始深度值 (mm)，uint16 格式
  - PNG: 16位灰度图
  - NPY: numpy 数组
- **depth_vis**: 伪彩色可视化，用于预览

### 元数据 (metadata.json)

```json
{
  "device": "Orbbec Gemini 335L",
  "serial_number": "CP2AB53000D6",
  "total_frames": 200,
  "duration_sec": 3.33,
  "streams": {
    "color": true,
    "depth": true,
    "ir_dual": true,
    "imu": true
  },
  "imu_records_count": 667
}
```

## 常见问题

### 1. 分辨率不匹配
**错误**: `depth/ir streams must have the same resolution and frame rate`

**解决**: 确保 `config.yaml` 中 depth 和 ir 的分辨率、帧率完全一致。

## 技术细节

### 激光控制
采集 IR 数据时，程序会自动关闭激光发射器，避免散斑干扰。退出时自动恢复。

### 帧同步
- 视频流使用 `FULL_FRAME_REQUIRE` 模式，确保 Color/Depth/IR 同步
- IMU 使用 `ANY_SITUATION` 模式，独立采集不受视频帧率限制

### 性能优化
- IMU 独立线程，10ms 轮询间隔
- 视频流 100ms timeout
- 支持无预览模式（`preview.enabled: false`）降低 CPU 占用

## 支持的设备

- Orbbec Gemini 335L
- 其他 Orbbec SDK v2 兼容设备（需测试验证）
