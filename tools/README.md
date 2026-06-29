# Orbbec SDK 工具集

所有诊断和工具脚本的统一索引。

## 📁 目录结构

```
tools/
├── camera_tools/           # 相机专用工具
│   ├── 335L_list_profiles.py
│   ├── 335L_get_params.py
│   ├── 305_list_profiles.py
│   └── 305_get_params.py
├── analyze_framerate.py    # 帧率分析
├── enumerate_profiles.py   # 枚举所有配置
├── find_common_profiles.py # 查找公共配置
├── get_camera_intrinsics.py # 获取内参（通用）
└── get_exposure_params.py  # 获取曝光参数
```

## 🔧 工具说明

### 相机专用工具 (camera_tools/)

#### Gemini 335L

**列出支持的配置**
```bash
python tools/camera_tools/335L_list_profiles.py
```
显示335L所有传感器支持的分辨率、帧率、格式。

**获取相机参数**
```bash
python tools/camera_tools/335L_get_params.py
```
输出内参、外参到 `camera_params.json`。

#### Gemini 305

**列出支持的配置**
```bash
python tools/camera_tools/305_list_profiles.py
```
显示305所有传感器支持的分辨率、帧率、格式。

**获取相机参数**
```bash
python tools/camera_tools/305_get_params.py
```
输出内参、外参、双目基线到 `camera_params.json`。

---

### 通用工具

#### enumerate_profiles.py - 枚举配置文件
```bash
python tools/enumerate_profiles.py
```
列出当前连接相机的所有流配置（Color/Depth/IR）。

#### find_common_profiles.py - 查找公共配置
```bash
python tools/find_common_profiles.py
```
查找Color和Depth都支持的分辨率+帧率组合，用于配置同步采集。

#### get_camera_intrinsics.py - 获取相机内参
```bash
python tools/get_camera_intrinsics.py
```
详细输出彩色和深度相机内参，保存到 `camera_intrinsics/`。

#### get_exposure_params.py - 获取曝光参数
```bash
python tools/get_exposure_params.py
```
获取当前相机的曝光时间、增益范围。

#### analyze_framerate.py - 帧率分析
```bash
python tools/analyze_framerate.py
```
实时分析采集帧率和延迟。

---

## 💡 使用建议

| 场景 | 推荐工具 |
|------|---------|
| 查看相机支持什么分辨率 | `camera_tools/{model}_list_profiles.py` |
| 配置多流同步 | `find_common_profiles.py` |
| 获取标定参数 | `camera_tools/{model}_get_params.py` 或 `get_camera_intrinsics.py` |
| 调试帧率问题 | `analyze_framerate.py` |
| 调整曝光参数 | `get_exposure_params.py` |

---

## 📌 注意

- 运行相机专用工具前，请确保对应型号的相机已连接
- 通用工具会自动检测连接的相机型号
- 部分工具输出JSON文件，可用于自动化配置
