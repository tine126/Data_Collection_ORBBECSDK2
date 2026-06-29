# Orbbec SDK 项目重构方案

## 项目概述

针对 **Gemini 335L** 和 **Gemini 305** 两款相机的代码重构方案。

---

## 目标目录结构

```
orbbecSDKv2/
├── cameras/
│   ├── gemini_335L/
│   │   ├── capture_rgb_stream.py       # 连续采集 RGB
│   │   ├── capture_rgbd_stream.py      # 连续采集 RGB-D
│   │   ├── capture_rgb_single.py       # 单帧采集 RGB
│   │   ├── capture_rgbd_single.py      # 单帧采集 RGB-D
│   │   ├── get_params.py               # 获取内外参
│   │   ├── list_profiles.py            # 列出相机支持的所有配置
│   │   ├── config.yaml                 # 335L 分辨率配置
│   │   └── README.md                   # 335L 使用说明
│   │
│   ├── gemini_305/
│   │   ├── capture_dual_rgb_stream.py  # 连续采集双路RGB
│   │   ├── capture_rgbd_stream.py      # 连续采集RGB-D
│   │   ├── capture_dual_rgb_single.py  # 单帧采集双路RGB
│   │   ├── capture_rgbd_single.py      # 单帧采集 RGB-D
│   │   ├── get_params.py               # 获取内外参
│   │   ├── list_profiles.py            # 列出相机支持的所有配置
│   │   ├── config.yaml                 # 305 分辨率配置
│   │   └── README.md                   # 305 使用说明
│   │
│   └── shared/
│       ├── frame_utils.py              # 帧转换工具
│       └── device_utils.py             # 设备检测工具
│
├── tools/                              # 通用工具
│   ├── diagnostics.py                  # 设备诊断工具（合并test/check脚本）
│   ├── analyze_framerate.py            # 帧率分析
│   ├── enumerate_profiles.py           # 配置枚举
│   └── find_common_profiles.py         # 查找公共配置
│
├── output/                             # 输出目录
├── captures/                           # 捕获输出目录
├── config.yaml                         # 全局配置模板
├── main.py                             # 主程序入口
├── REFACTORING_PLAN.md                 # 本文件
└── README.md                           # 项目说明
```

---

## 脚本功能详解

### Gemini 335L（6个核心脚本）

| 脚本名 | 功能说明 |
|--------|---------|
| `capture_rgb_stream.py` | 连续采集纯RGB流，支持实时预览、分辨率可配置 |
| `capture_rgbd_stream.py` | 连续采集RGB+深度流，支持对齐、实时预览 |
| `capture_rgb_single.py` | 快速单帧RGB采集，无预览，适合批量采集 |
| `capture_rgbd_single.py` | 快速单帧RGB-D采集，支持深度图可视化 |
| `get_params.py` | 获取相机内参、外参、畸变参数，输出JSON |
| `list_profiles.py` | 列出相机所有支持的分辨率、帧率、格式配置 |

**配置文件示例** (`cameras/gemini_335L/config.yaml`):
```yaml
# RGB分辨率配置
rgb:
  resolutions:
    - {width: 640, height: 480, fps: 30}
    - {width: 1280, height: 720, fps: 30}
    - {width: 1920, height: 1080, fps: 30}
  default: {width: 1280, height: 720, fps: 30}

# RGB-D 分辨率配置
rgbd:
  color: {width: 1280, height: 720, fps: 30}
  depth: {width: 640, height: 480, fps: 30}
  align: true  # 是否启用深度对齐到彩色
```

---

### Gemini 305（6个核心脚本）

| 脚本名 | 功能说明 |
|--------|---------|
| `capture_dual_rgb_stream.py` | 连续采集双路RGB流，支持左右相机同步预览 |
| `capture_rgbd_stream.py` | 连续采集RGB+深度流，支持对齐、实时预览 |
| `capture_dual_rgb_single.py` | 快速单帧双路RGB采集，输出左右图像 |
| `capture_rgbd_single.py` | 快速单帧RGB-D采集，支持深度图可视化 |
| `get_params.py` | 获取双目相机内参、外参、基线等参数 |
| `list_profiles.py` | 列出相机所有支持的分辨率、帧率、格式配置 |

**配置文件示例** (`cameras/gemini_305/config.yaml`):
```yaml
# 双路RGB配置
dual_rgb:
  left: {width: 1280, height: 800, fps: 30}
  right: {width: 1280, height: 800, fps: 30}
  sync: true  # 左右相机同步

# RGB-D 分辨率配置
rgbd:
  color: {width: 1280, height: 800, fps: 30}
  depth: {width: 640, height: 400, fps: 30}
  align: true
```

---

## 现有文件迁移映射

### Gemini 335L 相关文件

| 现有文件 | 目标位置 | 操作说明 |
|---------|---------|---------|
| `capture_335L_dual_rgb.py` | 拆分到 `cameras/gemini_335L/capture_rgb_stream.py` 和 `capture_rgbd_stream.py` | 提取RGB和RGB-D逻辑分离 |
| `test_335L_dual_rgb.py` | 合并到 `tools/diagnostics.py` | 整合为统一诊断工具 |
| `check_335L.py` | 合并到 `tools/diagnostics.py` | 整合为统一诊断工具 |

### Gemini 305 相关文件

| 现有文件 | 目标位置 | 操作说明 |
|---------|---------|---------|
| `capture_dual_rgb.py` | `cameras/gemini_305/capture_dual_rgb_stream.py` | 直接迁移并优化 |
| `capture_single_frame.py` | 拆分到两个相机的 `capture_*_single.py` | 按相机型号分离 |
| `capture_color_stereo_ir.py` | 整合到 `cameras/gemini_305/capture_rgbd_stream.py` | 提取RGB-D逻辑 |

### 工具脚本

| 现有文件 | 目标位置 | 操作说明 |
|---------|---------|---------|
| `get_dual_rgb_camera_params.py` | 拆分到各相机的 `get_params.py` | 按相机型号分离参数获取逻辑 |
| `list_camera_profiles.py` | `tools/list_profiles.py` | 重命名并保留 |
| `tools/get_camera_intrinsics.py` | 合并到各相机的 `get_params.py` | 功能整合 |

### 删除/整合

- **删除目录**: `dual_rgb/` - 内容整合到新结构
- **保留文件**: `main.py`, `config.yaml`, `README.md`

---

## 共享模块设计

### `cameras/shared/frame_utils.py`

提取公共的帧处理函数：
- `frame_to_bgr()` - 格式转换
- `save_frame()` - 保存图像
- `visualize_depth()` - 深度可视化

### `cameras/shared/device_utils.py`

提取公共的设备检测函数：
- `detect_camera_model()` - 检测相机型号
- `check_dual_rgb_support()` - 检查双RGB支持
- `get_sensor_list()` - 获取传感器列表

---

## 实施步骤

1. ✅ **创建目录结构**
   - 创建 `cameras/gemini_335L/`, `cameras/gemini_305/`, `cameras/shared/`
   - 创建配置文件模板

2. **提取共享代码**
   - 从现有脚本提取 `frame_to_bgr()` 等公共函数到 `shared/`

3. **迁移 Gemini 335L 脚本**
   - 基于 `capture_335L_dual_rgb.py` 创建5个脚本
   - 添加分辨率配置支持

4. **迁移 Gemini 305 脚本**
   - 基于 `capture_dual_rgb.py` 创建5个脚本
   - 添加分辨率配置支持

5. **整合工具脚本**
   - 合并测试/诊断脚本到 `tools/diagnostics.py`
   - 重命名并整理其他工具

6. **清理旧文件**
   - 删除根目录下的旧脚本
   - 删除 `dual_rgb/` 目录

7. **更新文档**
   - 更新 `README.md` 说明新结构
   - 为每个相机创建使用说明

---

## 优势

✅ **清晰的组织结构** - 按相机型号分类，功能一目了然  
✅ **易于维护** - 每个脚本职责单一，修改影响范围小  
✅ **配置化** - 分辨率等参数从代码分离到配置文件  
✅ **代码复用** - 共享模块减少重复代码  
✅ **扩展性强** - 新增相机型号只需添加新目录

---

## 注意事项

- 保持向后兼容：旧脚本在迁移完成前保留
- 配置文件格式统一使用 YAML
- 输出目录结构：`output/{camera_model}/{date}/{rgb|rgbd|dual_rgb}/`
- 所有脚本支持命令行参数覆盖配置文件

---

**文档创建时间**: 2026-06-29  
**作者**: Kiro AI Assistant
