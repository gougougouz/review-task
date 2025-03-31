import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_canny(img, kernel_size=5, sigma=1.0, low_threshold=50, high_threshold=150):
    """支持不同滤波器尺寸的Canny实现"""
    # 确保核大小为奇数
    kernel_size = kernel_size | 1
    
    # 1. 高斯滤波降噪
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # 2. 计算梯度幅值和方向（保持不变）
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle = np.mod(angle, 180)
    
    # 3. 非极大值抑制（保持不变）
    suppressed = np.zeros_like(magnitude)
    # ... [原非极大值抑制代码]
    
    # 4. 双阈值检测（保持不变）
    edges = np.zeros_like(suppressed)
    # ... [原双阈值代码]
    
    return edges.astype(np.uint8)

def process_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 测试4种不同滤波器尺寸
    kernel_sizes = [3, 5, 7, 9]
    results = []
    
    for ksize in kernel_sizes:
        edges = custom_canny(gray, kernel_size=ksize)
        results.append({
            'kernel': ksize,
            'edges': edges,
            'edge_pixels': np.sum(edges > 0)
        })
    
    # Hough检测使用默认5x5滤波结果
    edges_for_hough = custom_canny(gray, kernel_size=5)
    lines = cv2.HoughLines(edges_for_hough, 1, np.pi/180, 150)
    
    # 可视化布局
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 不同滤波尺寸的Canny结果
    for idx, res in enumerate(results, start=2):
        plt.subplot(2, 3, idx)
        plt.imshow(res['edges'], cmap='gray')
        plt.title(f'{res["kernel"]}x{res["kernel"]}滤波\n边缘像素：{res["edge_pixels"]}')
        plt.axis('off')
    
    # Hough直线检测结果
    result_img = img.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # ... [原直线绘制代码]
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Hough直线检测 (5x5滤波)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('canny_filter_compare.jpg', dpi=150)
    plt.show()

# 执行处理
process_image('input_image.jpg')