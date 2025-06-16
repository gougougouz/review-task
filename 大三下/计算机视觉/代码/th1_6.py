import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detector(img, kernel_size=5, sigma=1.0, low_thresh=50, high_thresh=150):
    """
    完整实现Canny边缘检测器
    """
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # 1. 高斯滤波
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # 2. 计算梯度
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
    edges[weak] = 75
    
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
            line_equations.append(f"x*{np.cos(theta):.3f} + y*{np.sin(theta):.3f} = {rho:.1f}")
    
    return line_equations

def visualize_process(img_path):
    """优化布局的可视化流程"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_sizes = [3, 5, 7]

    # 创建定制化布局
    fig = plt.figure(figsize=(30, 20))  # 增大画布尺寸
    gs = fig.add_gridspec(nrows=2, ncols=4,
                         height_ratios=[1.5, 1],   # 调整行比例
                         width_ratios=[1, 1, 1, 0.6],  # 调整列比例
                         hspace=0.2, wspace=0.1)  # 减少间距

    # 原始图像
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax0.set_title("Original Image (Enlarged)", fontsize=16, pad=20)
    ax0.axis('off')

    # 不同核尺寸的边缘检测
    for i, ksize in enumerate(kernel_sizes):
        ax = fig.add_subplot(gs[0, i+1])
        edges = canny_edge_detector(gray, kernel_size=ksize)
        ax.imshow(edges, cmap='gray')
        ax.set_title(f"Kernel {ksize}x{ksize}\nEdge Pixels: {np.sum(edges>0):,}", 
                    fontsize=14, pad=15)
        ax.axis('off')

    # Hough检测结果
    final_edges = canny_edge_detector(gray, kernel_size=5)
    line_equations = hough_line_detection(final_edges)
    
    ax1 = fig.add_subplot(gs[1, :3])
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    lines = cv2.HoughLines(final_edges, 1, np.pi/180, 150)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = (max(0, int(x0 - 1500*b)), max(0, int(y0 - 1500*a)))
            pt2 = (min(img.shape[1]-1, int(x0 + 1500*b)), 
                   min(img.shape[0]-1, int(y0 + 1500*a)))
            cv2.line(result_img, pt1, pt2, (255,0,0), 4, cv2.LINE_AA)  # 加粗线条
    
    ax1.imshow(result_img)
    ax1.set_title("Hough Line Detection with Extended Lines", fontsize=16, pad=20)
    ax1.axis('on')

    # 方程显示区域
    ax2 = fig.add_subplot(gs[1, 3])
    text_content = "Detected Line Equations:\n\n"
    for i, eq in enumerate(line_equations[:5]):  # 显示前5个方程
        text_content += f"Line {i+1}:\n{eq}\n\n"
    
    ax2.text(0.1, 0.5, text_content, 
            fontsize=14,
            family='monospace',
            linespacing=1.5,
            verticalalignment='center')
    ax2.axis('off')

    plt.savefig('optimized_layout.jpg', 
               dpi=300, 
               bbox_inches='tight',
               pad_inches=0.1)  # 减少保存时的留白
    print("优化结果已保存为 optimized_layout.jpg")
    plt.show()

if __name__ == "__main__":
    image_path = "D:/Samples/boudary.jpg"  # 请替换实际路径
    visualize_process(image_path)