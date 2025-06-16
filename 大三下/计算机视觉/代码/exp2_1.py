import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    """读取图像并转换颜色空间"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("图像读取失败，请检查路径")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def add_salt_pepper(img, prob=0.05):
    """椒盐噪声（双极性）"""
    noisy = np.copy(img)
    salt = np.random.rand(*img.shape[:2]) < prob/2
    pepper = np.random.rand(*img.shape[:2]) < prob/2
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

def add_impulse(img, prob=0.05, value=255):
    """脉冲噪声（单极性）"""
    noisy = np.copy(img)
    mask = np.random.rand(*img.shape[:2]) < prob
    noisy[mask] = value
    return noisy

def add_gaussian(img, mean=0, sigma=30):
    """高斯噪声"""
    row, col, ch = img.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_filters(img):
    """应用三种滤波器"""
    mean = cv2.blur(img, (5,5))
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    median = cv2.medianBlur(img, 5)
    return mean, gaussian, median

# 主程序
if __name__ == "__main__":
    # 读取图像（替换为您的路径）
    original = read_image(r"D:\Samples\cvImage.jpg")
    
    # 生成三种噪声
    salt_pepper = add_salt_pepper(original)
    impulse_white = add_impulse(original, value=255)
    gaussian_noise = add_gaussian(original)
    
    # 对每种噪声应用三种滤波器
    sp_mean, sp_gauss, sp_median = apply_filters(salt_pepper)
    iw_mean, iw_gauss, iw_median = apply_filters(impulse_white)
    gn_mean, gn_gauss, gn_median = apply_filters(gaussian_noise)
    
    # 可视化结果
    plt.figure(figsize=(18, 12))
    
    # 显示椒盐噪声处理结果
    images = [original, salt_pepper, sp_mean, sp_gauss, sp_median]
    titles = ['原图', '椒盐噪声', '均值滤波', '高斯滤波', '中值滤波']
    for i in range(5):
        plt.subplot(3, 5, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    
    # 显示白噪声处理结果
    images = [original, impulse_white, iw_mean, iw_gauss, iw_median]
    for i in range(5):
        plt.subplot(3, 5, i+6)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    
    # 显示高斯噪声处理结果
    images = [original, gaussian_noise, gn_mean, gn_gauss, gn_median]
    for i in range(5):
        plt.subplot(3, 5, i+11)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存所有结果（按实验要求命名）
    cv2.imwrite('cv01_学号_姓名_salt_pepper.jpg', cv2.cvtColor(salt_pepper, cv2.COLOR_RGB2BGR))
    cv2.imwrite('cv01_学号_姓名_impulse_white.jpg', cv2.cvtColor(impulse_white, cv2.COLOR_RGB2BGR))
    cv2.imwrite('cv01_学号_姓名_gaussian_noise.jpg', cv2.cvtColor(gaussian_noise, cv2.COLOR_RGB2BGR))
    # 保存其他处理结果...