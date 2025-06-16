import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path: str, convert_gray: bool = True) -> tuple:
    """
    加载并预处理图像
    :param image_path: 图像路径
    :param convert_gray: 是否转换为灰度图
    :return: (原始图像, 灰度图像/None)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if convert_gray else None
    return image, gray

def detect_edges(gray: np.ndarray, 
                threshold1: int = 50, 
                threshold2: int = 150) -> np.ndarray:
    """
    Canny边缘检测
    :param gray: 灰度图像
    :param threshold1: 低阈值
    :param threshold2: 高阈值
    :return: 边缘图像
    """
    return cv2.Canny(gray, threshold1, threshold2)

def hough_lines(edges: np.ndarray, 
               rho: float = 1, 
               theta: float = np.pi/180, 
               threshold: int = 100,
               min_line_length: int = 50,
               max_line_gap: int = 10) -> np.ndarray:
    """
    霍夫变换检测直线
    :param edges: 边缘图像
    :param rho: 距离分辨率
    :param theta: 角度分辨率
    :param threshold: 累加器阈值
    :param min_line_length: 最小线段长度
    :param max_line_gap: 最大线段间隙
    :return: 检测到的直线数组
    """
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)
    return lines if lines is not None else np.array([])

def hough_circles(gray: np.ndarray,
                 dp: float = 1,
                 min_dist: int = 20,
                 param1: int = 50,
                 param2: int = 30,
                 min_radius: int = 0,
                 max_radius: int = 0) -> np.ndarray:
    """
    霍夫变换检测圆
    :param gray: 灰度图像
    :param dp: 累加器分辨率
    :param min_dist: 圆之间的最小距离
    :param param1: Canny高阈值
    :param param2: 累加器阈值
    :param min_radius: 最小半径
    :param max_radius: 最大半径
    :return: 检测到的圆数组
    """
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                             param1=param1, param2=param2,
                             minRadius=min_radius, maxRadius=max_radius)
    return circles if circles is not None else np.array([])

def visualize_detections(image: np.ndarray,
                        lines: np.ndarray = None,
                        circles: np.ndarray = None) -> np.ndarray:
    """
    可视化检测结果
    :param image: 原始图像
    :param lines: 直线数组
    :param circles: 圆数组
    :return: 绘制结果的图像
    """
    result = image.copy()
    
    # 绘制直线
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 绘制圆
    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 外圆
            cv2.circle(result, (i[0], i[1]), 2, (255, 0, 0), 3)    # 圆心
    
    return result

def calculate_weld_angle(lines: np.ndarray) -> float:
    """
    计算焊缝角度
    :param lines: 直线数组
    :return: 角度(度)
    """
    if len(lines) < 2:
        raise ValueError("至少需要两条直线来计算角度")
    
    # 按线段长度排序
    line_lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) 
                   for [[x1, y1, x2, y2]] in lines]
    sorted_indices = np.argsort(line_lengths)[::-1]
    
    # 获取两条最长的直线
    line1 = lines[sorted_indices[0]][0]
    line2 = lines[sorted_indices[1]][0]
    
    # 计算角度
    angle1 = np.arctan2(line1[3]-line1[1], line1[2]-line1[0])
    angle2 = np.arctan2(line2[3]-line2[1], line2[2]-line2[0])
    
    # 计算最小夹角
    angle_diff = np.abs(angle1 - angle2)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    
    return np.rad2deg(angle_diff)

def plot_hough_space(edges: np.ndarray, 
                    theta_range: np.ndarray = np.linspace(-np.pi/2, np.pi/2, 180)) -> None:
    """
    绘制霍夫空间投票器
    :param edges: 边缘图像
    :param theta_range: 角度范围
    """
    h, w = edges.shape
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    
    accumulator = np.zeros((len(rhos), len(theta_range)), dtype=np.uint64)
    
    y_idxs, x_idxs = np.nonzero(edges)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for j, theta in enumerate(theta_range):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, j] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(accumulator, cmap='hot', aspect='auto',
              extent=[np.rad2deg(theta_range[0]), np.rad2deg(theta_range[-1]),
                      rhos[-1], rhos[0]])
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.title('Hough Space Accumulator')
    plt.colorbar(label='Votes')
    plt.show()

def main():
    # 1. 加载图像
    image_path = "D:\Samples/bucket2.png"  # 替换为你的图像路径
    image, gray = load_image(image_path)
    
    # 2. 边缘检测
    edges = detect_edges(gray, threshold1=50, threshold2=150)
    
    # 3. 直线检测
    lines = hough_lines(edges, rho=1, theta=np.pi/180, threshold=100)
    
    # 4. 圆检测
    circles = hough_circles(gray, dp=1, min_dist=20, param1=50, param2=30)
    
    # 5. 可视化结果
    result_image = visualize_detections(image, lines, circles)
    
    # 6. 计算焊缝角度
    try:
        weld_angle = calculate_weld_angle(lines)
        print(f"焊缝角度: {weld_angle:.2f}°")
        
        # 在图像上标注角度
        cv2.putText(result_image, f"Weld Angle: {weld_angle:.2f}°", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except ValueError as e:
        print(f"角度计算失败: {str(e)}")
    
    # 7. 显示结果
    plt.figure(figsize=(12, 8))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Results'), plt.axis('off')
    plt.show()
    
    # 8. 显示霍夫空间
    plot_hough_space(edges)
    
    # 9. 保存结果
    cv2.imwrite('hough_result.jpg', result_image)
    print("结果已保存为 hough_result.jpg")

if __name__ == "__main__":
    main()