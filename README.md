# YOLO11n 轻量部署 (CPU)

## 硬件说明

- **CPU**: AMD Ryzen 7 8745HS — 完全支持，8核16线程，跑 YOLO11n 毫无压力
- **核显**: Radeon 780M (RDNA3) — ROCm 不支持桌面 iGPU，所以用 CPU 推理
- **ONNX Runtime + OpenCV DNN**: 纯 CPU 推理，不需要 CUDA/ROCM

## 环境（已就绪）

与 Qwen3-ASR 共用 conda 环境 **`D:\Anaconda\envs\yolo-asr`**（Python 3.12），
依赖已装好（ultralytics 8.4.x / onnx / opencv-python / numpy）。
`E:\YOLO\.venv` 是早期尝试留下的空环境，可忽略。

常用入口（先执行一次 `cd E:\YOLO; .\env.ps1` 可设好缓存目录）：
```powershell
& D:\Anaconda\envs\yolo-asr\python.exe E:\YOLO\detect.py 图片.jpg
& D:\Anaconda\envs\yolo-asr\python.exe E:\YOLO\detect_camera.py
```

## 模型

- 权重 `yolo11n.pt`（~5.6MB）已下载，ONNX 已导出到 `models\yolo11n.onnx`（~10.1MB）。
- 重新导出：`& D:\Anaconda\envs\yolo-asr\python.exe export_model.py`
  （会自动从 GitHub/HF 官方源拉权重；导出需 onnxsim/onnxslim，已安装）
- 国内网络不可达时，可手动下载 `yolo11n.pt` 放到项目根目录，或临时配置镜像。

## 使用方法

```powershell
# 单张图片检测
& D:\Anaconda\envs\yolo-asr\python.exe detect.py image.jpg

# 保存结果到指定路径
& D:\Anaconda\envs\yolo-asr\python.exe detect.py image.jpg -o output.jpg

# 批量检测整个文件夹
& D:\Anaconda\envs\yolo-asr\python.exe detect.py images_folder/

# 摄像头实时检测
& D:\Anaconda\envs\yolo-asr\python.exe detect_camera.py

# 调整参数
& D:\Anaconda\envs\yolo-asr\python.exe detect.py image.jpg -c 0.3 -s 640   # 降低置信度阈值，提高输入尺寸
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-c / --conf` | 0.45 | 置信度阈值 |
| `-s / --size` | 320 | 输入尺寸 (越小越快) |
| `-m / --model` | models/yolo11n.onnx | 模型路径 |

## 模型信息

- **模型**: YOLO11n (Nano)
- **参数量**: ~2.6M
- **ONNX 文件**: ~10.1MB（simplify 后）
- **输入尺寸**: 320×320 (可选 320/416/640)
- **类别**: COCO 80 类
- **实测**: 8745HS CPU 约 17ms/帧（320×320，bus.jpg 检出 5 目标）

## 文件结构

```
E:\YOLO\
├── models/
│   └── yolo11n.onnx        # 导出的模型（已就绪）
├── requirements.txt        # 依赖
├── export_model.py         # 模型导出脚本
├── detect.py               # 图片检测
├── detect_camera.py        # 摄像头实时检测
├── setup.bat               # 一键安装（优先用共用 conda 环境）
├── env.ps1                 # 共用环境辅助脚本（设缓存目录 + 常用命令）
└── README.md               # 说明文档
```
