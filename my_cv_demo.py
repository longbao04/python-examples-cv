"""摄像头基础图像处理示例：运行 python3 my_cv_demo.py -c 0。"""

import argparse

import cv2


def main():
    # 与项目其他示例保持一致：-c 指定摄像头编号，默认使用 0。
    parser = argparse.ArgumentParser(description="灰度、模糊、边缘与轮廓检测示例")
    parser.add_argument(
        "-c", "--camera_to_use", type=int, default=0,
        help="摄像头编号（默认：0）",
    )
    args = parser.parse_args()

    # 直接使用 OpenCV 读取摄像头，便于初学者理解处理流程。
    cap = cv2.VideoCapture(args.camera_to_use)
    try:
        if not cap.isOpened():
            print("无法打开摄像头，请检查编号、连接和摄像头权限。")
            return 1

        # 窗口标题使用英文，避免部分系统无法正确显示中文。
        original_window = "Original + Largest Contour"
        edges_window = "Canny Edges"
        cv2.namedWindow(original_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(edges_window, cv2.WINDOW_NORMAL)
        print("请选中任意图像窗口，按 x 退出。")

        while True:
            # 每次读取一帧；读取失败时退出，避免处理空图像。
            success, frame = cap.read()
            if not success or frame is None:
                print("无法读取摄像头画面，程序结束。")
                return 1

            # 1. 灰度转换：将 BGR 彩色图转换为单通道灰度图。
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. 高斯模糊：使用 5×5 的核减少噪声，核尺寸必须为正奇数。
            # 最后的 0 表示让 OpenCV 根据核尺寸自动计算标准差。
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. Canny 边缘检测：低、高阈值分别为 50 和 150。
            # 可调整这两个值，观察检测出的边缘如何变化。
            edges = cv2.Canny(blurred, 50, 150)

            # 4. 从边缘图中提取轮廓：只查找外部轮廓，并压缩冗余点。
            # 使用副本，保证显示的边缘图不受旧版 OpenCV 的修改影响。
            result = cv2.findContours(
                edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # OpenCV 3 返回三个值，OpenCV 4 返回两个值；倒数第二项都是轮廓。
            contours = result[-2]

            # 在原图副本上绘制面积最大的轮廓；没有轮廓时直接显示原图。
            display = frame.copy()
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # OpenCV 的颜色顺序为 BGR，(0, 255, 0) 表示绿色。
                cv2.drawContours(display, [largest_contour], -1, (0, 255, 0), 2)

            # 仅显示两个窗口；灰度图与模糊图作为中间处理结果。
            cv2.imshow(original_window, display)
            cv2.imshow(edges_window, edges)

            # waitKey 同时刷新窗口并读取按键；0xFF 提取按键码的低 8 位。
            if cv2.waitKey(1) & 0xFF == ord("x"):
                break

        return 0
    finally:
        # 正常退出或出现异常时，都释放摄像头并关闭窗口。
        cap.release()
        cv2.destroyAllWindows()


# 只有直接运行本文件时才启动摄像头，导入本文件不会启动。
if __name__ == "__main__":
    raise SystemExit(main())
