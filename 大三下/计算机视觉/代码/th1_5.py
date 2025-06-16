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
    """均衡布局优化"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read the image:{img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 处理参数
    kernel_sizes = [3, 5, 7]
    
    # 创建平衡布局（2行3列）
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(nrows=2, ncols=3,
                        height_ratios=[1.2, 1],  # 调整行高比例
                        width_ratios=[1, 1, 1.2],  # 右侧留更多空间
                        hspace=0.3, wspace=0.25)
    
    # 原始图像（第一行左）
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax0.set_title("Original Image ", fontsize=12)
    ax0.axis('off')
    
    # 边缘检测结果（第一行中、右）
    for i, ksize in enumerate(kernel_sizes[:2]):
        ax = fig.add_subplot(gs[0, i+1])
        edges = canny_edge_detector(gray, kernel_size=ksize)
        ax.imshow(edges, cmap='gray')
        ax.set_title(f"Kernel {ksize}x{ksize}\nEdge Pixels: {np.sum(edges>0):,}", fontsize=11)
        ax.axis('off')
     # 第三个边缘检测结果（第二行左）
    ax2 = fig.add_subplot(gs[1, 0])
    edges = canny_edge_detector(gray, kernel_size=7)
    ax2.imshow(edges, cmap='gray')
    ax2.set_title(f"Kernel 7x7\nEdges: {np.sum(edges>0):,}", fontsize=11)
    ax2.axis('off')
    
    
    # Hough检测结果（第二行中）
    ax1 = fig.add_subplot(gs[1, 1])
    final_edges = canny_edge_detector(gray, kernel_size=5)
    line_equations = hough_line_detection(final_edges)
    
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, 150)
    if lines is not None:
        for line in lines[:]:  # 仅处理前5条直线
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = (max(0, int(x0 - 1200*b)), max(0, int(y0 - 1200*a)))
            pt2 = (min(img.shape[1]-1, int(x0 + 1200*b)), 
                   min(img.shape[0]-1, int(y0 + 1200*a)))
            cv2.line(result_img, pt1, pt2, (255,0,0), 2, cv2.LINE_AA)
    
    ax1.imshow(result_img)
    ax1.set_title("Hough Lines Visualization ", fontsize=12)
    ax1.axis('on')
    
   
    
    # 方程显示区域（第二行右）
    ax3 = fig.add_subplot(gs[1, 2])
    text_content = "Top 5 Line Equations:\n\n"
    for i, eq in enumerate(line_equations[:5]):  # 显示前5个方程
        text_content += f"{i+1}. {eq}\n\n"
    
    ax3.text(0.05, 0.5, text_content, 
            fontsize=11,
            family='monospace',
            verticalalignment='center')
    ax3.axis('off')
    
    # 保存优化结果
    plt.savefig('balanced_result.jpg', 
               dpi=300, 
               bbox_inches='tight',
               pad_inches=0.15)
    print("优化结果已保存为 balanced_result.jpg")
    plt.show()

# 使用示例
if __name__ == "__main__":
    image_path = "D:\Samples\eage1.jpg"  
    visualize_process(image_path)