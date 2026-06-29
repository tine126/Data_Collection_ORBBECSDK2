# Gemini 335L 使用指南

## 脚本说明

### 1. 连续RGB采集
```bash
python cameras/gemini_335L/capture_rgb_stream.py
```
- 实时预览RGB流
- 按空格开始/停止采集
- 按Q退出

### 2. 连续RGB-D采集
```bash
python cameras/gemini_335L/capture_rgbd_stream.py
```
- 同时采集彩色和深度
- 实时显示深度可视化
- 按空格开始/停止，Q退出

### 3. 单帧RGB采集
```bash
python cameras/gemini_335L/capture_rgb_single.py
```
快速采集单帧RGB图像

### 4. 单帧RGB-D采集
```bash
python cameras/gemini_335L/capture_rgbd_single.py
```
快速采集单帧RGB+深度图像

### 5. 获取相机参数
```bash
python cameras/gemini_335L/get_params.py
```
输出内参、外参到 `camera_params.json`

### 6. 列出支持的配置
```bash
python cameras/gemini_335L/list_profiles.py
```
显示所有支持的分辨率和帧率

## 配置

编辑 `config.yaml` 修改分辨率、帧率、曝光等参数。

## 输出目录

所有采集文件保存到 `output/gemini_335L/`
