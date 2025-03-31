import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detector(img, kernel_size=5, sigma=1.0, low_thresh=50, high_thresh=150):
    """
    完整实现Canny边缘检测器
    参数：
    - kernel_size: 高斯核尺寸（必须为奇数）
    - sigma: 高斯标准差
    - low_thresh/high_thresh: 滞后阈值
    """
    # 确保核为奇数
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # 1. 抑制噪声 - 高斯滤波
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # 2. 计算梯度大小和方向
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    direction = np.mod(direction, 180)
    
    # 3. 非极大值抑制
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            angle = direction[i,j]
            # 确定相邻像素位置
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i,j-1], magnitude[i,j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
            else:
                neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
            
            if magnitude[i,j] >= max(neighbors):
                suppressed[i,j] = magnitude[i,j]
    
    # 4. 滞后阈值
    edges = np.zeros_like(suppressed)
    strong = suppressed > high_thresh
    weak = (suppressed >= low_thresh) & (suppressed <= high_thresh)
    
    edges[strong] = 255
    edges[weak] = 75  # 标记弱边缘
    
    # 5. 连通性分析
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if edges[i,j] == 75:
                if np.any(edges[i-1:i+2, j-1:j+2] == 255):
                    edges[i,j] = 255
                else:
                    edges[i,j] = 0
    
    return edges.astype(np.uint8)

def hough_line_detection(edges):
    """Hough直线检测与方程提取"""
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    line_equations = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # 转换为直角坐标系方程：x*cosθ + y*sinθ = ρ
            line_equations.append(f"x*{np.cos(theta):.3f} + y*{np.sin(theta):.3f} = {rho:.1f}")
    
    return line_equations

def visualize_process(img_path):
    """完整处理流程可视化"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read the image:{img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 处理参数
    kernel_sizes = [3, 5, 7]
    
    # 可视化布局
    plt.figure(figsize=(18, 12))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # 不同核尺寸的边缘检测
    for idx, ksize in enumerate(kernel_sizes, start=2):
        edges = canny_edge_detector(gray, kernel_size=ksize)
        plt.subplot(2, 3, idx)
        plt.imshow(edges, cmap='gray')
        plt.title(f"kernel size {ksize}x{ksize}\nEdge Pixels:{np.sum(edges>0)}")
        plt.axis('off')
    
    # Hough检测结果
    final_edges = canny_edge_detector(gray, kernel_size=5)
    line_equations = hough_line_detection(final_edges)
    
    # 绘制直线
    result_img = img.copy()
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, 150)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
            cv2.line(result_img, pt1, pt2, (0,0,255), 2)
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Hough linear detection")
    plt.axis('off')
    
    # 显示直线方程
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.3, "The equation of the detected line:\n" + "\n".join(line_equations[:3]), 
             fontsize=12, family='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('canny_full_process.jpg', dpi=150)
    plt.show()

# 使用示例
if __name__ == "__main__":
    image_path = "D:/Samples/boudary.jpg"  # 替换为实际路径
    visualize_process(image_path)