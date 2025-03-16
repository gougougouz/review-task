import cv2
import numpy
# 1. 读取图片


image = cv2.imread("D:/Samples/demo1.BMP", cv2.IMREAD_COLOR)

# 检查图片是否成功读取
if image is None:
    print("图片读取失败，请检查路径是否正确！")
    exit()

# 2. 显示图片
# 使用 cv2.imshow() 显示图片

cv2.imshow("Displayed Image", image)

# 等待按键
cv2.waitKey(0)

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()

# 3. 保存图片

cv2.imwrite("saved_image.jpg", image)
print("图片已保存为 saved_image.jpg")