"""
YOLO11n ONNX 推理脚本 - CPU 轻量版
用法:
  python detect.py image.jpg              # 单张图片
  python detect.py image.jpg -o out.jpg   # 保存结果
  python detect.py images_folder/         # 批量检测
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ── 模型路径：基于脚本所在目录解析，避免在不同 CWD 下找不到模型 ──
MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "yolo11n.onnx")

# ── COCO 80 类名 ──────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

# ── 颜色 ──────────────────────────────────────────────────
COLORS = np.random.RandomState(42).randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def load_model(model_path: str, input_size: int = 320):
    """加载 ONNX 模型"""
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    print(f"[INFO] 模型加载完成: {model_path}  输入尺寸: {input_size}x{input_size}")
    return net, input_size


def preprocess(img: np.ndarray, input_size: int):
    """预处理: letterbox resize → blob"""
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1 / 255.0,
        size=(input_size, input_size),
        swapRB=True, crop=False,
    )
    return blob


def postprocess(img: np.ndarray, outputs: np.ndarray, conf_thres: float = 0.45,
                iou_thres: float = 0.5, input_size: int = 320):
    """
    后处理: NMS + 坐标还原
    outputs shape: (1, 84, 400) for 320x320 → 400 = 20*20 + 10*10 + 5*5
    """
    h, w = img.shape[:2]
    preds = outputs[0]  # (84, N)

    # 转置为 (N, 84)
    preds = preds.T

    # 取 class scores (前4列是 xywh, 后80列是 class scores)
    boxes = preds[:, :4]
    class_scores = preds[:, 4:]

    # 每个框取最大类别分数
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # 过滤低置信度
    mask = max_scores > conf_thres
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    # xywh → xyxy (中心点格式转左上右下)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # 还原到原图坐标
    x1 = (x1 / input_size * w).astype(int)
    y1 = (y1 / input_size * h).astype(int)
    x2 = (x2 / input_size * w).astype(int)
    y2 = (y2 / input_size * h).astype(int)

    # 裁剪到图像范围
    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)

    # NMS
    dets = np.column_stack([x1, y1, x2, y2, max_scores])
    indices = cv2.dnn.NMSBoxes(
        [[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])] for i in range(len(dets))],
        list(max_scores),
        conf_thres,
        iou_thres,
    )

    if len(indices) == 0:
        return [], [], []

    # indices 可能是嵌套数组
    if isinstance(indices, np.ndarray) and indices.ndim > 1:
        indices = indices.flatten()

    final_boxes = [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])) for i in indices]
    final_scores = [float(max_scores[i]) for i in indices]
    final_ids = [int(class_ids[i]) for i in indices]

    return final_boxes, final_scores, final_ids


def draw_results(img, boxes, scores, class_ids):
    """在图像上绘制检测结果"""
    for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids):
        label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
        color = tuple(int(c) for c in COLORS[cid])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def detect_image(net, input_size: int, image_path: str, conf_thres: float = 0.45,
                 save_path: str = None):
    """单张图片检测"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return

    # 预处理
    blob = preprocess(img, input_size)
    net.setInput(blob)

    # 推理
    t0 = time.time()
    outputs = net.forward()
    elapsed = time.time() - t0

    # 后处理
    boxes, scores, class_ids = postprocess(img, outputs, conf_thres=conf_thres,
                                           input_size=input_size)

    # 绘制
    draw_results(img, boxes, scores, class_ids)

    # 输出信息
    print(f"\n{'='*50}")
    print(f"文件: {image_path}")
    print(f"耗时: {elapsed*1000:.1f} ms (CPU)")
    print(f"检测到 {len(boxes)} 个目标:")
    for score, cid in zip(scores, class_ids):
        label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
        print(f"  - {label}: {score:.2%}")
    print(f"{'='*50}")

    # 保存或显示
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"[INFO] 结果已保存: {save_path}")
    else:
        out_name = f"result_{Path(image_path).name}"
        cv2.imwrite(out_name, img)
        print(f"[INFO] 结果已保存: {out_name}")

    return boxes, scores, class_ids


def main():
    parser = argparse.ArgumentParser(description="YOLO11n ONNX CPU 轻量检测")
    parser.add_argument("input", help="输入图片路径或文件夹")
    parser.add_argument("-m", "--model", default=MODEL_PATH, help="ONNX 模型路径")
    parser.add_argument("-o", "--output", default=None, help="输出路径")
    parser.add_argument("-s", "--size", type=int, default=320, help="输入尺寸 (默认320)")
    parser.add_argument("-c", "--conf", type=float, default=0.45, help="置信度阈值 (默认0.45)")
    args = parser.parse_args()

    net, input_size = load_model(args.model, args.size)

    input_path = Path(args.input)
    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        print(f"[INFO] 找到 {len(images)} 张图片")
        for img_path in images:
            detect_image(net, input_size, str(img_path), conf_thres=args.conf)
    else:
        detect_image(net, input_size, str(input_path), conf_thres=args.conf,
                     save_path=args.output)


if __name__ == "__main__":
    main()
