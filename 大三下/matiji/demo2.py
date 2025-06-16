import numpy as np
import matplotlib.pyplot as plt

# 设置参数
np.random.seed(123)
days = 250
base_level = 0.8  # 基础位置(0-1之间，1为最顶部)
volatility = 0.05

# 生成高位随机数据
random_walk = np.cumsum(np.random.normal(0, volatility, days))
values = base_level + 0.15 * random_walk  # 确保值在0.7-0.9范围内

# 创建图表
plt.figure(figsize=(12, 6), facecolor='white')

# 设置坐标范围，保留下方空白
plt.ylim(0, 1)  # y轴范围0-1
plt.xlim(0, days)

# 绘制折线（调整线条样式）
plt.plot(values, 
         color='#1f77b4',  # 蓝色线条
         linewidth=2,
         alpha=0.9)

# 完全移除坐标轴和边框
plt.axis('off')

# 调整边距使折线悬浮在高位
plt.subplots_adjust(bottom=0.4)  # 增加底部空白

plt.savefig('output.png', format='png', dpi=300)