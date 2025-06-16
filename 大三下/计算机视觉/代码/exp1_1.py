import cv2

# 读取图像的函数
def read_image(image_path):
    """
    读取图像
    :param image_path: 图像文件的路径
    :return: 返回读取的图像(numpy数组)
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: 无法读取图像，请检查路径是否正确。")
    return img

# 写入图像的函数
def write_image(image_path, img):
    """
    写入图像
    :param image_path: 保存图像的路径
    :param img: 要保存的图像(numpy数组)
    :return: 返回是否保存成功
    """
    success = cv2.imwrite(image_path, img)
    if not success:
        print("Error: 无法保存图像，请检查路径和格式是否正确。")
    return success