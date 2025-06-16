import numpy as np
import matplotlib.pyplot as plt

# 设置参数
np.random.seed(11233)  # 更改随机种子以生成不同数据
days = 250
base_level = 0.85  # 调整基础位置，使整体更高
volatility = 0.07  # 增加波动幅度，使波动稍微大一些

# 生成高位随机数据
random_walk = np.cumsum(np.random.normal(0, volatility, days))
values = base_level + 0.22 * random_walk  # 调整系数，确保值在合理范围内

# 创建图表
plt.figure(figsize=(12, 6), facecolor='white')

# 设置坐标范围，保留下方空白
plt.ylim(0, 1)  # y轴范围0-1
plt.xlim(0, days)

# 绘制折线（调整线条样式）
plt.plot(values, 
         color='#70AD47',  # 橙色线条
         linewidth=2,
         alpha=0.9)

# 完全移除坐标轴和边框
plt.axis('off')

# 调整边距使折线悬浮在高位
plt.subplots_adjust(bottom=0.4)  # 增加底部空白

plt.savefig('output_modified_larger_fluctuation.png', format='png', dpi=300)
plt.show()  # 显示图表