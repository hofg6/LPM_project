import cv2
import numpy as np
import serial
import time
import json
import function
from function import order_points, adjust_exposure, multi_frame_denoise


class CalibrationSystem:
    def __init__(self, width, height):
        self.calibration_data = self.load_calibration()
        self.width = width
        self.height = height
        self.frame_buffer = []  # 新增多帧缓冲区
        self.buffer_size = 5  # 缓冲帧数（根据工频周期调整）

    def update_buffer(self, frame):
        """维护固定大小的环形缓冲区"""
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame.copy())

    def load_calibration(self):
        try:
            with open('calibration.json', 'r') as f:
                data = json.load(f)
                data['M'] = np.array(data['M'])  # 恢复矩阵结构
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save_calibration(self, screen_corners):
        # 计算透视变换矩阵
        if screen_corners is not None and len(screen_corners) == 4:
            ordered_corners = order_points(screen_corners)
            target_points = np.array([[0, 0], [self.width - 1, 0],
                                      [self.width - 1, self.height - 1], [0, self.height - 1]], dtype=np.float32)
            m = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), target_points)

            calibration_data = {
                "screen_corners": ordered_corners.tolist(),
                "M": m.tolist(),
                "resolution": [1920, 1080]
            }
            with open('calibration.json', 'w') as f:
                json.dump(calibration_data, f, indent=2)


def calibrate_screen(frame):
    """
    :parameter:标定屏幕位置
    :param frame: 图像数组
    :return:屏幕角点坐标+frame
    """
    frame2 = frame.copy()
    # 图像二值化
    frame = adjust_exposure(frame2, 6)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, screen_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel)

    # 检测轮廓(基于面积假设: 屏幕是场景中最大的亮区)
    contours, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找物体外轮廓
    max_contour = max(contours, key=cv2.contourArea) if contours else None  # 依据面积选择最大轮廓

    screen_corners = None
    if max_contour is not None:
        # 方案1：动态调整epsilon检测四边形
        epsilon_factor = 0.02
        max_attempts = 5
        for _ in range(max_attempts):  # 自适应调整逼近精度(0.02<=epsilon_factor<=0.07)
            epsilon = epsilon_factor * cv2.arcLength(max_contour, True)  # 逼近精度参数
            approx = cv2.approxPolyDP(max_contour, epsilon, True)  # 多边形逼近以减少多边形的点数
            if len(approx) == 4:
                # cv2.approxPolyDP输入:(N, 1, 2);reshape(4,2)输出:(N, 2),便于后续计算
                screen_corners = order_points(approx.reshape(4, 2))
                break
            else:
                epsilon_factor += 0.01
        else:
            # 方案2：最小外接矩形
            rect = cv2.minAreaRect(max_contour)  # 返回最小外接矩形,rect:Box2D结构[（中心（x，y）,（宽度，高度）,旋转角度）]
            box = cv2.boxPoints(rect)  # 通过矩形中心、宽高、旋转角度，确定矩形四个顶点坐标
            screen_corners = order_points(box)

            # 方案3：凸包+最远点
            if screen_corners is None:
                hull = cv2.convexHull(max_contour)  # 凸包消除凹陷区域(将最外层的点连接起来构成的凸多边形)
                approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)  # 多边形逼近以减少多边形的点数
                if len(approx) >= 4:
                    screen_corners = function.farthest_points(approx.reshape(-1, 2), 4)  # 最远点选择确保角点分散
                else:
                    # 保底：使用全图边界
                    # 当无法检测到有效屏幕区域时的保底方案，定义整张图像的全幅区域作为默认屏幕区域
                    h, w = frame.shape[:2]  # 获取图像的高度，宽度
                    screen_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    ordered_corners = order_points(screen_corners)
    for point in ordered_corners:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 0), -1)  # 画圆标记矩形位置

    return screen_corners, frame  # 屏幕坐标


def process_image(frame, screen_corners, m):
    """
    :parameter:主处理函数
    :param frame:输入图像
    :param screen_corners:屏幕坐标
    :param m:变换矩阵
    :return:frame screen_x screen_y
    """
    frame_height, frame_width, _ = frame.shape
    target_screen_width = 1920
    target_screen_height = 1080
    screen_x, screen_y = 0, 0
    ordered_corners = order_points(screen_corners)
    flag_point = False
    for point in ordered_corners:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 0), -1)  # 画圆标记矩形位置
    # 激光点检测预处理
    frame = adjust_exposure(frame, 5)
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red1 = np.array([0, 100, 100])  # H:0-10, 高饱和度+亮度
    # upper_red1 = np.array([10, 255, 255])
    lower_red1 = np.array([0, 0, 100])  # H:0-10, 高饱和度+亮度
    upper_red1 = np.array([40, 255, 255])
    hsv_mask = cv2.inRange(hsv_mask, lower_red1, upper_red1)

    #  形态学操作
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # 大核膨胀合并碎片
    hsv_mask = cv2.dilate(hsv_mask, kernel_dilate, iterations=1)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)  # 开操作去噪
    # 激光点定位与映射
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 外轮廓检测
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # 面积过滤小噪声
            perimeter = cv2.arcLength(contour, True)  # 计算周长
            if perimeter == 0:
                continue
            # 圆形度公式：(4π*面积)/(周长²)，完美圆=1
            circularity = 4 * np.pi * area / (perimeter ** 2)  # 圆形度计算
            if circularity < 0.5:  # 过滤非圆形区域
                continue
            cx, cy = function.refine_point(frame, contour)  # 亚像素级定位
            # 透视变换映射
            if m is not None:
                point = np.array([[[cx, cy]]], dtype=np.float32)  # 将二维坐标点转换为符合cv2.perspectiveTransform的输入格式要求
                transformed_point = cv2.perspectiveTransform(point, m)  # 透视变换
                screen_x = int(transformed_point[0][0][0])
                screen_y = int(transformed_point[0][0][1])
                # 坐标边界保护
                screen_x = max(0, min(screen_x, target_screen_width - 1))
                screen_y = max(0, min(screen_y, target_screen_height - 1))
            # 降级方案：线性映射
            else:
                # 获得一个图像的最小矩形边框
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(screen_corners) if screen_corners is not None else (
                    0, 0, frame_width, frame_height)
                # 通过计算归一化坐标，通过乘以目标分辨率得到物理坐标
                screen_x = int((cx - x_rect) / w_rect * target_screen_width)
                screen_y = int((cy - y_rect) / h_rect * target_screen_height)
            if screen_x != 0 and screen_y != 0:
                flag_point = True
                # 可视化标记
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                cv2.putText(frame, f"({screen_x}, {screen_y})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame, screen_x, screen_y, flag_point


def main():
    ser = serial.Serial('COM6', 9600, timeout=1)
    time.sleep(2)  # 等待串口稳定
    screen_width, screen_height = 1920, 1080
    cal_sys = CalibrationSystem(screen_width, screen_height)
    cap = cv2.VideoCapture(1)
    # 相机参数设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 禁用自动曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 初始为手动曝光，范围根据硬件决定(-1~-13)
    cap.set(cv2.CAP_PROP_FPS, 60)
    f_note = False
    f_calibrate = False
    screen_corners = None
    now = 0
    last_send = 0
    release_Mouse = [0x57, 0xAB, 0x00, 0x04, 0x07, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f]
    r_Mouse = [0x57, 0xAB, 0x00, 0x04, 0x07, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10]
    m = None
    # ser.write(bytes(r_Mouse))
    # time.sleep(0.1)
    try:
        while True:
            ret, avg_frame = cap.read()
            if not ret:
                return
            # # 维护多帧缓冲区
            # cal_sys.update_buffer(frame)
            # # 执行多帧平均降噪
            # if len(cal_sys.frame_buffer) >= cal_sys.buffer_size:
            #     avg_frame = multi_frame_denoise(cal_sys.frame_buffer)
            # else:
            #     avg_frame = frame  # 缓冲区未满时使用原始帧
            cmd = function.check_serial_command(ser)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 禁用自动曝光
            cap.set(cv2.CAP_PROP_EXPOSURE, -7)
            # 判断串口接受到的命令
            if cmd == 'start':  # 进入笔记模式
                f_note = True
                m = cal_sys.calibration_data['M']
                screen_corners = cal_sys.calibration_data['screen_corners']
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 禁用自动曝光
                cap.set(cv2.CAP_PROP_EXPOSURE, -7)
                print("开始笔记模式")
            elif cmd == 'stop':  # 清除笔记
                f_note = False
                if cv2.getWindowProperty('Laser Pointer Tracking', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Laser Pointer Tracking')
                print("退出笔记模式")
            elif cmd == 'calibrate':  # 标定模式
                f_calibrate = True
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 自动曝光
                print("开始标定模式")
            elif cmd == 'success_calibrate' and f_calibrate:
                if screen_corners is not None:
                    f_calibrate = False
                    cal_sys.save_calibration(screen_corners)
                    print(screen_corners)
                    print("标定成功！")
                else:
                    print("标定失败：未检测到有效四边形")
                if cv2.getWindowProperty('S_Calibration', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('S_Calibration')
            # 标记模式
            if f_calibrate:
                screen_corners, avg_frame = calibrate_screen(avg_frame)
                cv2.imshow('S_Calibration', avg_frame)
            # 笔记模式
            if f_note:
                # cv2.imshow('Original', avg_frame)
                frame, screen_x, screen_y, flag_point = process_image(avg_frame, screen_corners, m)
                cv2.imshow('Laser Pointer Tracking', frame)
                if flag_point:
                    data_packet = function.note_mov(screen_width, screen_height, int(screen_x), int(screen_y))
                    ser.write(bytes(data_packet))
                    time.sleep(0.01)
                    print(f"MouseMovLocation: {[f'0x{b:02X}' for b in data_packet]}")

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by User")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        ser.close()


if __name__ == '__main__':
    main()
