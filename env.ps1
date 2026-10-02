# ============================================================
#  yolo-asr 共用环境辅助脚本
#  用途: 把 pip / HuggingFace / Ultralytics 的缓存全部指到
#        E:\YOLO\.tmp（不污染 C 盘），并给出常用命令。
#  说明: 包源使用官方 PyPI；模型权重从 Hugging Face 下载
#        （ASR 也可用 ModelScope 备用：--source modelscope）。
#  用法:  cd E:\YOLO ; .\env.ps1     （在当前终端内生效）
# ============================================================

$env:PIP_CACHE_DIR    = "E:\YOLO\.tmp\pipcache"
$env:HF_HOME          = "E:\YOLO\.tmp\hf-cache"
$env:MODELSCOPE_CACHE = "E:\YOLO\.tmp\modelscope-cache"
$env:YOLO_CONFIG_DIR  = "E:\YOLO\.tmp\yolo-config"

# conda 环境 python 完整路径（不依赖 conda activate 即可使用）
$PyEnv = "D:\Anaconda\envs\yolo-asr\python.exe"

Write-Host ""
Write-Host "== yolo-asr 环境已就绪 ==" -ForegroundColor Cyan
Write-Host ("python : {0}" -f $PyEnv)
Write-Host ""
Write-Host "常用命令（复制即用）：" -ForegroundColor Green
Write-Host ("  [YOLO] 图片检测      & {0} E:\YOLO\detect.py 图片.jpg" -f $PyEnv)
Write-Host ("  [YOLO] 导出ONNX模型  & {0} E:\YOLO\export_model.py" -f $PyEnv)
Write-Host ("  [ASR]  下载模型权重  & {0} E:\YOLO\qwen3-asr\download_model.py --model qwen3-asr-0.6b" -f $PyEnv)
Write-Host ("  [ASR]  环境检查      & {0} E:\YOLO\qwen3-asr\check_env.py" -f $PyEnv)
Write-Host ("  [ASR]  启动服务       & {0} E:\YOLO\qwen3-asr\server.py" -f $PyEnv)
Write-Host ""
