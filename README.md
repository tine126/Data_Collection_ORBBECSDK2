# Orbbec SDK v2 数据采集工具使用文档

## 项目结构

```
orbbecSDKv2/
├── cameras/                    # 相机脚本（按型号分类）
│   ├── gemini_335L/           # Gemini 335L 专用脚本
│   ├── gemini_305/            # Gemini 305 专用脚本
│   └── shared/                # 共享工具模块
├── tools/                     # 通用工具脚本
├── output/                    # 输出目录
├── config.yaml               # 全局配置（参考）
└── main.py                   # 主程序（传统采集）
```

## 环境要求

- Python 3.8+
- 依赖：`pyorbbecsdk2`, `opencv-python`, `numpy<2`, `pyyaml`

## 安装依赖

```bash
conda activate env  #自己的conda环境
pip install pyyaml opencv-python "numpy<2"
pip install pyorbbecsdk2
```

## 快速开始

### Gemini 335L
```bash
# 查看支持的配置
python cameras/gemini_335L/list_profiles.py

# 连续采集 RGB-D
python cameras/gemini_335L/capture_rgbd_stream.py

# 获取相机参数
python cameras/gemini_335L/get_params.py
```

详见 [cameras/gemini_335L/README.md](cameras/gemini_335L/README.md)

### Gemini 305
```bash
# 查看支持的配置
python cameras/gemini_305/list_profiles.py

# 连续采集双路RGB
python cameras/gemini_305/capture_dual_rgb_stream.py

# 获取相机参数
python cameras/gemini_305/get_params.py
```

详见 [cameras/gemini_305/README.md](cameras/gemini_305/README.md)

---

## 完整使用指南

### 1. 基础采集流程

#### 步骤 1：查看相机支持的配置

在开始采集前，先查看相机支持哪些分辨率和帧率：

**Gemini 335L:**
```bash
python cameras/gemini_335L/list_profiles.py
```

**Gemini 305:**
```bash
python cameras/gemini_305/list_profiles.py
```

输出示例：
```
Sensor: OBSensorType.COLOR_SENSOR
  [0] 640x480 @ 30fps | OBFormat.MJPG
  [1] 1280x720 @ 30fps | OBFormat.MJPG
  [2] 1920x1080 @ 30fps | OBFormat.MJPG
```

#### 步骤 2：修改配置文件（可选）

编辑 `cameras/{camera_model}/config.yaml` 修改分辨率、曝光等参数：

```yaml
rgb:
  default: {width: 1280, height: 720, fps: 30}
  format: "MJPG"
  auto_exposure: false
  exposure: 10000
  gain: 100
```

#### 步骤 3：运行采集脚本

选择合适的采集模式运行脚本（详见下文各模式说明）。

---

### 2. Gemini 335L 使用详解

#### 2.1 连续 RGB 流采集
```bash
python cameras/gemini_335L/capture_rgb_stream.py
```

**功能：**
- 实时预览彩色图像
- 按空格键开始/停止录制
- 按Q或ESC退出

**输出：**
- 保存路径：`output/gemini_335L/rgb_stream/{timestamp}/`
- 文件格式：`frame_000001.png`, `frame_000002.png`, ...

**使用场景：** 纯彩色数据采集、视频录制、相机测试

#### 2.2 连续 RGB-D 流采集
```bash
python cameras/gemini_335L/capture_rgbd_stream.py
```

**功能：**
- 同时采集彩色和深度
- 实时显示深度可视化（伪彩色）
- 支持深度对齐到彩色（D2C）

**输出：**
- 彩色图：`color_000001.png`
- 深度图：`depth_000001.png`（16位PNG，单位mm）

**使用场景：** 3D重建、SLAM、深度估计

#### 2.3 单帧 RGB 快速采集
```bash
python cameras/gemini_335L/capture_rgb_single.py
```

**功能：** 无预览窗口，快速采集一帧彩色图像

**输出：** `output/gemini_335L/rgb_single/rgb_{timestamp}.png`

**使用场景：** 批量采集、定时任务、快速测试

#### 2.4 单帧 RGB-D 快速采集
```bash
python cameras/gemini_335L/capture_rgbd_single.py
```

**功能：** 快速采集一帧RGB+深度

**输出：**
- `color_{timestamp}.png`
- `depth_{timestamp}.png`

**使用场景：** 快速标定、单帧3D采集

#### 2.5 获取相机参数
```bash
python cameras/gemini_335L/get_params.py
```

**功能：** 获取相机内参、畸变参数

**输出：** `cameras/gemini_335L/camera_params.json`

```json
{
  "color": {
    "width": 1280, "height": 720,
    "fx": 911.23, "fy": 910.45,
    "cx": 640.12, "cy": 360.34,
    "distortion": [0.01, -0.02, ...]
  },
  "depth": {...}
}
```

**使用场景：** 相机标定、3D坐标转换

---

### 3. Gemini 305 使用详解

#### 3.1 连续双路 RGB 流采集
```bash
python cameras/gemini_305/capture_dual_rgb_stream.py
```

**功能：**
- 同时采集左右RGB相机
- 实时预览左右图像（横向拼接）
- 自动切换到双RGB模式

**输出：**
- 左相机：`left_000001.png`
- 右相机：`right_000001.png`

**使用场景：** 双目立体视觉、深度估计、3D重建

#### 3.2 连续 RGB-D 流采集
```bash
python cameras/gemini_305/capture_rgbd_stream.py
```

同335L的RGB-D采集，支持彩色+深度同步。

#### 3.3 单帧双路 RGB 采集
```bash
python cameras/gemini_305/capture_dual_rgb_single.py
```

快速采集一帧双目图像。

#### 3.4 单帧 RGB-D 采集
```bash
python cameras/gemini_305/capture_rgbd_single.py
```

快速采集一帧RGB-D。

#### 3.5 获取双目参数
```bash
python cameras/gemini_305/get_params.py
```

**输出：** 包含左右相机内参、外参、基线距离

**使用场景：** 双目标定、立体匹配

---

### 4. 高级功能：完整多流采集

对于需要同时采集 **Color + Depth + IR + IMU** 的场景，使用 `main.py`：

```bash
python main.py
```

或指定配置文件：
```bash
python main.py -c config.yaml
```

#### 快捷键

| 按键 | 功能 |
|------|------|
| **U** | 自动录制200帧后停止 |
| **SPACE** | 手动开始/停止录制 |
| **S** | 保存单帧快照 |
| **Q / ESC** | 退出程序 |

#### 输出目录结构
```
output/capture_YYYYMMDD_HHMMSS/
├── color/           # 彩色图
├── depth_raw/       # 原始深度（16位PNG或npy）
├── depth_vis/       # 深度可视化
├── ir_left/         # 左红外
├── ir_right/        # 右红外
├── imu_data.csv     # IMU数据（加速度+陀螺仪）
└── metadata.json    # 采集元数据
```

**使用场景：** SLAM数据集采集、传感器融合、科研数据集

---

### 5. 工具脚本使用

所有工具脚本位于 `tools/` 目录，详见 [tools/README.md](tools/README.md)

#### 常用工具

**查找多流公共配置：**
```bash
python tools/find_common_profiles.py
```
输出Color和Depth都支持的分辨率组合，用于配置同步采集。

**获取详细内参：**
```bash
python tools/get_camera_intrinsics.py
```
比相机目录的 `get_params.py` 更详细，输出到 `camera_intrinsics/`。

**帧率分析：**
```bash
python tools/analyze_framerate.py
```
实时监控采集帧率和延迟。

---

### 6. 配置文件说明

#### 相机专用配置

每个相机目录都有独立的 `config.yaml`：

**Gemini 335L** (`cameras/gemini_335L/config.yaml`)：
```yaml
rgb:
  default: {width: 1280, height: 720, fps: 30}
  format: "MJPG"
  auto_exposure: false
  exposure: 10000      # 曝光时间 (μs)
  gain: 100           # 增益

rgbd:
  color: {width: 1280, height: 720, fps: 30}
  depth: {width: 640, height: 480, fps: 30}
  align: true         # 深度对齐到彩色
  min_depth_mm: 20
  max_depth_mm: 10000
  colormap: "JET"     # 深度可视化色彩映射

output:
  base_dir: "../../output/gemini_335L"
  color_format: "png"
  depth_format: "png"
```

**Gemini 305** (`cameras/gemini_305/config.yaml`)：
```yaml
dual_rgb:
  left: {width: 1280, height: 800, fps: 30}
  right: {width: 1280, height: 800, fps: 30}
  sync: true          # 左右同步
  
rgbd:
  color: {width: 1280, height: 800, fps: 30}
  depth: {width: 640, height: 400, fps: 30}
  align: true
```

#### 全局配置 (config.yaml)

用于 `main.py` 的完整多流采集配置，包含IMU、对齐模式等高级参数。

---

### 7. 常见使用场景

#### 场景 1：快速测试相机是否正常
```bash
# 查看支持配置
python cameras/gemini_335L/list_profiles.py

# 采集单帧
python cameras/gemini_335L/capture_rgb_single.py
```

#### 场景 2：采集双目立体图像对
```bash
# 连续采集左右RGB
python cameras/gemini_305/capture_dual_rgb_stream.py
# 按空格开始录制，采集100对图像后按空格停止
```

#### 场景 3：采集RGB-D数据用于3D重建
```bash
# 配置分辨率（可选）
vim cameras/gemini_335L/config.yaml

# 连续采集RGB-D
python cameras/gemini_335L/capture_rgbd_stream.py
```

#### 场景 4：获取标定参数
```bash
# 方法1：快速获取
python cameras/gemini_335L/get_params.py

# 方法2：详细输出
python tools/get_camera_intrinsics.py
```

#### 场景 5：SLAM数据集采集（含IMU）
```bash
# 编辑config.yaml启用所需数据流
python main.py
# 按U自动录制200帧
```

---

### 8. 输出数据说明

#### 深度数据格式

**depth_*.png** - 16位灰度PNG
- 像素值单位：毫米（mm）
- 读取方式：
```python
import cv2
depth = cv2.imread('depth_000001.png', cv2.IMREAD_UNCHANGED)
# depth[y, x] 即为该点深度值（mm）
```

#### IMU数据格式 (imu_data.csv)

| 字段 | 单位 | 说明 |
|------|------|------|
| sys_ts | 秒 | 系统时间戳 |
| accel_x/y/z | m/s² | 加速度 |
| gyro_x/y/z | rad/s | 角速度 |

---

### 9. 故障排查

#### 问题1：找不到相机
```
错误：No device found
```
**解决：**
1. 检查USB连接（推荐USB 3.0）
2. 确认相机供电正常
3. Windows：检查设备管理器
4. Linux：检查 `lsusb` 是否识别

#### 问题2：分辨率不支持
```
错误：Failed to start pipeline
```
**解决：**
1. 运行 `list_profiles.py` 查看支持的配置
2. 修改 `config.yaml` 使用支持的分辨率

#### 问题3：numpy版本错误
```
错误：ValueError: ndarray is not C-contiguous
```
**解决：**
```bash
pip install "numpy<2"
```

#### 问题4：帧率低/卡顿
**解决：**
1. 降低分辨率
2. 关闭预览窗口（修改配置 `preview.enabled: false`）
3. 使用USB 3.0接口
4. 运行 `tools/analyze_framerate.py` 诊断

---

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

### 分辨率、帧率和像素格式
```yaml
color:
  width: 640
  height: 480
  fps: 30
  format: "MJPG"   # 像素格式: MJPG, RGB, BGR, YUYV, NV12 等（留空则自动匹配）

depth:
  width: 640
  height: 480
  fps: 30
  format: "Y16"    # 像素格式: Y16
  min_depth_mm: 20
  max_depth_mm: 10000
  colormap: "JET"  # 可选: JET, TURBO, INFERNO, BONE, HOT

ir:
  width: 848
  height: 480
  fps: 30
  format: "Y16"    # 像素格式: Y8, Y16
```

**注意**：
- 推荐 Color/Depth 使用 640x480@30fps，支持硬件 D2C 对齐
- 硬件 D2C 对齐（`ALIGN_D2C_HW_MODE`）要求 Color 和 Depth 分辨率相同
- 可通过 `tools/enumerate_profiles.py` 查看设备支持的所有格式和分辨率

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

### 2. numpy 版本过高导致深度数据报错
**错误**: `ValueError: ndarray is not C-contiguous`

**原因**: numpy 2.0+ 对内存连续性要求更严格，与 pyorbbecsdk 不兼容。

**解决**: 降级 numpy 到 1.x：
```bash
pip install "numpy<2"
```

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
