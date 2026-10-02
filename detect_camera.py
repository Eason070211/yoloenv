"""
YOLO11n ONNX 摄像头实时检测 - CPU 版
用法:
  python detect_camera.py                    # 默认摄像头
  python detect_camera.py --camera 1         # 第二个摄像头
  python detect_camera.py --size 320         # 输入尺寸
"""
import argparse
import time

import cv2
import numpy as np

from detect import load_model, preprocess, postprocess, draw_results, MODEL_PATH, COCO_CLASSES, COLORS


def main():
    parser = argparse.ArgumentParser(description="YOLO11n ONNX CPU 摄像头实时检测")
    parser.add_argument("-m", "--model", default=MODEL_PATH, help="ONNX 模型路径")
    parser.add_argument("-s", "--size", type=int, default=320, help="输入尺寸")
    parser.add_argument("-c", "--conf", type=float, default=0.45, help="置信度阈值")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--width", type=int, default=640, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=480, help="摄像头高度")
    args = parser.parse_args()

    net, input_size = load_model(args.model, args.size)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("[INFO] 按 'q' 退出，按 's' 截图保存")

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 无法读取帧")
                break

            # 预处理 + 推理
            blob = preprocess(frame, input_size)
            net.setInput(blob)

            t0 = time.time()
            outputs = net.forward()
            infer_time = time.time() - t0

            # 后处理
            boxes, scores, class_ids = postprocess(frame, outputs,
                                                   conf_thres=args.conf,
                                                   input_size=input_size)

            # 绘制
            draw_results(frame, boxes, scores, class_ids)

            # FPS 计算
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()

            # 显示 FPS
            cv2.putText(frame, f"FPS: {fps_display:.1f} | Inference: {infer_time*1000:.0f}ms",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Objects: {len(boxes)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("YOLO11n - CPU Detection (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[INFO] 截图已保存: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
