import cv2
import pyrealsense2 as rs
import numpy as np

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 创建一个对齐对象
align = rs.align(rs.stream.color)

image_counter = 1  # 用于跟踪保存的图像数量

try:
    while True:
        # 等待RealSense相机准备好数据
        frames = pipeline.wait_for_frames()

        # 获取彩色图像和深度图像
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 转换为OpenCV格式
        color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())

        # 将深度图像从黑色到白色映射
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=2.0, beta=0)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255), cv2.COLORMAP_BONE)
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=65535.0 / 1)

        # 显示彩色图像
        cv2.imshow("Color Image", color_image)

        # 显示深度图像
        cv2.imshow("Depth Image", depth_colormap)

        # 检测按下的键
        key = cv2.waitKey(1)

        # 如果按下's'键，保存彩色图像和深度图像
        if key == ord('s'):
            color_filename = f"{image_counter}r.png"  # 彩色图像文件名
            depth_filename = f"{image_counter}d.tiff"  # 深度图像文件名
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)
            image_counter += 1  # 更新图像计数

        # 如果按下'q'键，退出预览
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
