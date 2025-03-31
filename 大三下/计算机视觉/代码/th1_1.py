import cv2
import numpy as np
import matplotlib.pyplot as plt

# 实现Canny边缘检测器
def custom_canny(img, sigma=1.0, low_threshold=50, high_threshold=150):
    """自定义Canny边缘检测实现"""
    # 1. 高斯滤波降噪
    blurred = cv2.GaussianBlur(img, (5,5), sigma)
    
    # 2. 计算梯度幅值和方向
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算幅值和方向
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle = np.mod(angle, 180)  # 转换为0-180度
    
    # 3. 非极大值抑制
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            dir = angle[i,j]
            if (0 <= dir < 22.5) or (157.5 <= dir <= 180):
                neighbors = [magnitude[i,j-1], magnitude[i,j+1]]
            elif 22.5 <= dir < 67.5:
                neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
            elif 67.5 <= dir < 112.5:
                neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
            else:
                neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
            
            if magnitude[i,j] >= max(neighbors):
                suppressed[i,j] = magnitude[i,j]
    
    # 4. 双阈值检测和边缘连接
    edges = np.zeros_like(suppressed)
    strong_edges = suppressed > high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    
    edges[strong_edges] = 255
    edges[weak_edges] = 75  # 标记为弱边缘
    
    # 边缘连接
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if edges[i,j] == 75:
                if np.any(edges[i-1:i+2,j-1:j+2] == 255):
                    edges[i,j] = 255
                else:
                    edges[i,j] = 0
    
    return edges.astype(np.uint8)

# 图像处理流程
def process_image(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny边缘检测
    edges = custom_canny(gray)
    
    # Hough直线检测
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    # 绘制检测到的直线
    result = img.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 2)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny边缘检测')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Hough直线检测')
    plt.axis('off')
    
    # 显示直线方程
    plt.subplot(224)
    plt.text(0.1, 0.5, "检测到的直线方程：\n" + 
             "\n".join([f"ρ={l[0][0]:.2f}, θ={l[0][1]*180/np.pi:.2f}°" 
                      for l in lines[:3]]), 
             fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cv_edge_detection.jpg')
    plt.show()

# 执行处理
process_image('input_image.jpg')