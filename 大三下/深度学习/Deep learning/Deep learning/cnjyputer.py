import torch
from torch import nn
from torch.nn import functional as F

print("创建第一个顺序模型")
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(f"模型结构: {net}")

X = torch.rand(2, 20)
print(f"输入数据形状: {X.shape}")
output = net(X)
print(f"输出结果形状: {output.shape}")
print(f"输出结果前5个元素: {output[0, :5]}\n")

print("=" * 50)
print("创建自定义MLP类")
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层
        print(f"初始化MLP: 隐藏层大小={self.hidden.weight.shape}, 输出层大小={self.out.weight.shape}")

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        hidden_output = self.hidden(X)
        print(f"隐藏层输出形状: {hidden_output.shape}")
        activated = F.relu(hidden_output)
        print(f"激活后形状: {activated.shape}")
        final_output = self.out(activated)
        print(f"最终输出形状: {final_output.shape}")
        return final_output
    
net = MLP()
print("调用MLP模型进行前向传播")
output = net(X)
print(f"MLP输出结果前5个元素: {output[0, :5]}\n")

print("=" * 50)
print("创建自定义Sequential类")
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module
            print(f"添加模块 {idx}: {module}")

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        print(f"MySequential输入形状: {X.shape}")
        for i, block in enumerate(self._modules.values()):
            X = block(X)
            print(f"经过模块 {i} 后的形状: {X.shape}")
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("调用MySequential模型进行前向传播")
output = net(X)
print(f"MySequential输出结果前5个元素: {output[0, :5]}\n")

print("=" * 50)
print("创建具有固定权重的MLP")
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        print(f"创建固定随机权重，形状: {self.rand_weight.shape}")
        self.linear = nn.Linear(20, 20)
        print(f"创建共享线性层，权重形状: {self.linear.weight.shape}")

    def forward(self, X):
        print(f"FixedHiddenMLP输入形状: {X.shape}")
        X = self.linear(X)
        print(f"第一次线性变换后形状: {X.shape}")
        
        # 使用创建的常量参数以及relu和mm函数
        mm_result = torch.mm(X, self.rand_weight)
        print(f"矩阵乘法后形状: {mm_result.shape}")
        X = F.relu(mm_result + 1)
        print(f"ReLU激活后形状: {X.shape}")
        
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        print(f"第二次线性变换后形状: {X.shape}")
        
        # 控制流
        sum_before = X.abs().sum().item()
        print(f"绝对值之和: {sum_before}")
        
        scaling_count = 0
        while X.abs().sum() > 1:
            X /= 2
            scaling_count += 1
        
        print(f"缩放次数: {scaling_count}，缩放后绝对值之和: {X.abs().sum().item()}")
        result = X.sum()
        print(f"最终输出(标量): {result.item()}")
        return result

net = FixedHiddenMLP()
print("调用FixedHiddenMLP模型进行前向传播")
output = net(X)
print(f"FixedHiddenMLP输出结果: {output.item()}\n")

print("=" * 50)
print("创建嵌套MLP")
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        print(f"创建内部Sequential网络: {self.net}")
        self.linear = nn.Linear(32, 16)
        print(f"创建输出线性层: {self.linear}")

    def forward(self, X):
        print(f"NestMLP输入形状: {X.shape}")
        net_output = self.net(X)
        print(f"内部网络输出形状: {net_output.shape}")
        final_output = self.linear(net_output)
        print(f"最终输出形状: {final_output.shape}")
        return final_output

print("创建混合模型Chimera")
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(f"Chimera模型结构: {chimera}")

print("调用Chimera模型进行前向传播")
output = chimera(X)
print(f"Chimera最终输出: {output.item()}")
