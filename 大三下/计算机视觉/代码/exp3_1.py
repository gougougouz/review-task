import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像预处理：读取图像并去噪
def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊去噪
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

# Sobel 边缘检测
def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(sobel)

# Prewitt 边缘检测
def prewitt_edge_detection(image):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)  # 转为浮点型
    prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)  # 转为浮点型
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
    return np.uint8(prewitt)  # 转回 uint8 以便显示

# Roberts 边缘检测
def roberts_edge_detection(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    roberts_x = cv2.filter2D(image, -1, kernel_x)
    roberts_y = cv2.filter2D(image, -1, kernel_y)
    roberts = cv2.magnitude(roberts_x, roberts_y)
    return np.uint8(roberts)

# Canny 边缘检测
def canny_edge_detection(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

# 显示结果
def display_results(original, results, titles):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    for i, (result, title) in enumerate(zip(results, titles)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 图像路径
    image_path = "D:\Samples\IMG_20231227_180043.jpg"

    # 图像预处理
    preprocessed_image = preprocess_image(image_path)

    # 各种边缘检测算法
    sobel_result = sobel_edge_detection(preprocessed_image)
    prewitt_result = prewitt_edge_detection(preprocessed_image)
    roberts_result = roberts_edge_detection(preprocessed_image)
    canny_result = canny_edge_detection(preprocessed_image, 50, 150)

    # 显示结果
    results = [sobel_result, prewitt_result, roberts_result, canny_result]
    titles = ["Sobel", "Prewitt", "Roberts", "Canny"]
    display_results(preprocessed_image, results, titles)

if __name__ == "__main__":
    main()