@echo off
echo ==========================================
echo   YOLO11n 轻量部署 - 一键安装
echo ==========================================
echo.

cd /d "%~dp0"

rem 优先使用共用 conda 环境 D:\Anaconda\envs\yolo-asr（YOLO + Qwen3-ASR 共用）
set "CONDA_PY=D:\Anaconda\envs\yolo-asr\python.exe"
if exist "%CONDA_PY%" (
    echo [1/3] 使用共用 conda 环境: %CONDA_PY%
    set "PY=%CONDA_PY%"
) else (
    echo [1/3] 未找到 conda 环境，创建本地 .venv ...
    if not exist ".venv" python -m venv .venv
    set "PY=.venv\Scripts\python.exe"
)

echo.
echo [2/3] 安装依赖（官方 PyPI）...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt

echo.
echo [3/3] 导出模型（首次运行自动下载 YOLO11n 权重）...
"%PY%" export_model.py

echo.
echo ==========================================
echo   安装完成！
echo   使用方法:
echo     "%PY%" detect.py your_image.jpg
echo     "%PY%" detect_camera.py
echo ==========================================
pause
