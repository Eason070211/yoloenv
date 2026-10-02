"""
导出 YOLO11n 为 ONNX 格式（仅需运行一次）
YOLO11n 是目前 Ultralytics 最轻量的模型 (~2.6MB)
"""
import os
from pathlib import Path
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"
ONNX_NAME = "yolo11n.onnx"
EXPORT_DIR = Path(__file__).parent / "models"

def main():
    EXPORT_DIR.mkdir(exist_ok=True)
    onnx_path = EXPORT_DIR / ONNX_NAME

    if onnx_path.exists():
        print(f"[SKIP] {onnx_path} 已存在，跳过导出。如需重新导出请删除该文件。")
        return

    print("[1/3] 下载 YOLO11n 模型...")
    model = YOLO(MODEL_NAME)  # 自动下载 yolo11n.pt

    print("[2/3] 导出为 ONNX (opset=17, simplify=True)...")
    model.export(
        format="onnx",
        imgsz=320,       # 输入尺寸 320x320，速度优先
        opset=17,
        simplify=True,
    )

    # ultralytics 导出到当前目录，移动到 models/
    exported = Path(MODEL_NAME.replace(".pt", ".onnx"))
    if exported.exists():
        exported.rename(onnx_path)
        print(f"[3/3] 模型已保存: {onnx_path}")
    else:
        print(f"[!] 导出文件未在预期位置找到: {exported}")
        print(f"    请手动将 {ONNX_NAME} 移动到 {EXPORT_DIR}")

if __name__ == "__main__":
    main()
