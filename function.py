import cv2
import numpy as np


def adjust_exposure(frame, gamma=1.0):
    """
    :parameters:调节曝光度
    :param frame: 图像数组
    :param gamma:<1提高曝光，>1降低曝光
    :return:调整曝光后的图像
    """
    # 生成伽马校正的查找表（Look-Up Table, LUT）
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)  # 应用查找表到输入图像


def multi_frame_denoise(buffer):
    """多帧平均降噪（兼容灰度/彩色图）"""
    if len(buffer) == 0:
        return None

    # 转换为浮点类型避免溢出
    avg_frame = np.mean([frame.astype(np.float32) for frame in buffer], axis=0)
    return avg_frame.astype(np.uint8)  # 转回uint8


def order_points(pts):
    """
    parameters:四边形顶点排序
    :param pts:形状为 (4, 2) 的NumPy数组，表示4个无序的二维点坐标
    :return:排序后的 (4, 2) 数组，顺序为 [左上, 右上, 右下, 左下]
    """
    rect = np.zeros((4, 2), dtype="float32")  # 返回来一个给定形状和类型的用0填充的数组
    pts = np.array(pts)
    s = pts.sum(axis=1)  # 计算每个点的 x + y 值（axis=0对应最外层的[]，axis=1对应第二外层的[]，…）
    rect[0] = pts[np.argmin(s)]  # 左上角，靠近图像原点
    rect[2] = pts[np.argmax(s)]  # 右下角，远离图像原点
    diff = np.diff(pts, axis=1)  # 计算每个点的 x - y 值
    rect[1] = pts[np.argmin(diff)]  # 右上角，x相对小，y相对大
    rect[3] = pts[np.argmax(diff)]  # 左下角，x相对大，y相对小
    return rect


def farthest_points(points, k=4):
    """从点集中选择距离最远的k个点"""
    centroids = points.mean(axis=0)  # 通过对所有点的x和y坐标分别求平均值，得到中心点坐标(axis=0 表示纵轴平均，输出的是格式（1，x）的格式)
    distances = np.linalg.norm(points - centroids, axis=1)  # 计算每个点到中心的欧式距离(多维空间中两个点的真实距离)
    idx = np.argsort(distances)[-k:]  # 通过对距离排序后取最后k个索引，确保选择的是最外围的点
    return points[idx]


def refine_point(frame, contour):
    """
    :parameter:亚像素级定位
    :param frame:原始BGR图像
    :param contour:激光点轮廓
    :return: (cx, cy): 优化后的整数坐标
    """
    # 计算图像矩，得到整数坐标（像素级精度）
    M = cv2.moments(contour)
    # 质心公式
    cx = int(M['m10'] / M['m00'])  # 轮廓区域的水平坐标质心
    cy = int(M['m01'] / M['m00'])  # 轮廓区域的垂直坐标质心

    # 亚像素优化准备
    # cv2.TERM_CRITERIA_EPS：当精度达到阈值（0.001）时停止。
    # cv2.TERM_CRITERIA_MAX_ITER：最多迭代30次。
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 组合条件：满足任一条件即终止迭代
    corners = np.array([[[cx, cy]]], dtype=np.float32)  # 转化为np.float32格式，且形状为（1,1,2）

    cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners, (5, 5), (-1, -1), criteria)  # 亚像素优化
    cx, cy = corners[0][0]
    return int(cx), int(cy)


def note_mov(width, height, screen_x, screen_y):
    """
        :parameter:生成绝对鼠标数据
        :return:绝对鼠标数据包
        """
    mouse_base = [0x57, 0xAB, 0x00, 0x04, 0x07, 0x02, 0x01]
    # 边界保护，避免程序崩溃
    screen_x = max(0, min(screen_x, width - 1))
    screen_y = max(0, min(screen_y, height - 1))
    x_cur = int((4096 * screen_x) / width)  # 计算x的下传值
    y_cur = int((4096 * screen_y) / height)  # 计算y的下传值
    xl_cur = x_cur & 0XFF  # 取x的低八位
    yl_cur = y_cur & 0XFF  # 取y的低八位
    xh_cur = (x_cur >> 8) & 0XFF  # 取x的高八位
    yh_cur = (y_cur >> 8) & 0XFF  # 取y的高八位
    data_sum = (0X10 + xl_cur + xh_cur + yl_cur + yh_cur) & 0XFF  # 计算累计和
    mouse_end = [xl_cur, xh_cur, yl_cur, yh_cur, 0x00, data_sum]
    return mouse_base + mouse_end


def check_serial_command(ser):
    """
        :parameter:检测串口指令是否匹配开始/停止条件
        :return: 'start' 开始指令 | 'stop' 停止指令 | None 无匹配
    """
    if ser.in_waiting >= 14:  # 检查缓冲区是否有足够数据
        data = ser.read(14)
        print(data)
        ser.flushInput()
        # 转换为十六进制列表便于比较
        hex_data = [b for b in data]

        # 笔记模式指令
        start_cmd = [0x57, 0xAB, 0x00, 0x02, 0x08, 0x01, 0x00,
                     0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20]
        # 清除笔记指令
        stop_cmd = [0x57, 0xAB, 0x00, 0x02, 0x08, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C]
        calib_cmd = [0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
                     0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7]
        success_calib_cmd = [0xDF, 0xDE, 0xDD, 0xDC, 0xDB, 0xDA, 0xD9,
                             0xD8, 0xD7, 0xD6, 0xD5, 0xD4, 0xD3, 0xD2]
        if hex_data == start_cmd:
            return 'start'
        elif hex_data == stop_cmd:
            return 'stop'
        elif hex_data == calib_cmd:
            return 'calibrate'
        elif hex_data == success_calib_cmd:
            return 'success_calibrate'
    return None
