import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 设置环境变量解决OpenMP错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义带有内部Softmax的模型
class SoftmaxModelWithActivation(nn.Module):
    def __init__(self):
        super(SoftmaxModelWithActivation, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3*32*32, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)  # 在模型内部应用Softmax
        return x

# 定义不带Softmax的模型（将在损失函数中使用）
class SoftmaxModelWithoutActivation(nn.Module):
    def __init__(self):
        super(SoftmaxModelWithoutActivation, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3*32*32, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)  # 直接返回logits，不应用Softmax
        return x

# 训练函数
def train_model(model, trainloader, testloader, criterion, optimizer, model_name, num_epochs=10):
    model.to(device)
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 对于带Softmax的模型，直接使用outputs计算准确率
            # 对于不带Softmax的模型，需要先计算最大值的索引
            if isinstance(model, SoftmaxModelWithActivation):
                _, predicted = torch.max(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
                
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # 计算平均训练损失
        train_loss = 0.0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
        
        train_losses.append(train_loss / len(trainloader))
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # 同样，根据模型类型决定如何计算预测结果
                if isinstance(model, SoftmaxModelWithActivation):
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        test_losses.append(test_loss / len(testloader))
        
        print(f'Epoch {epoch+1} - {model_name} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.2f}%')
    
    return train_losses, test_losses, train_accs, test_accs

# 绘制训练曲线
def plot_comparison(model1_results, model2_results, title1, title2):
    train_losses1, test_losses1, train_accs1, test_accs1 = model1_results
    train_losses2, test_losses2, train_accs2, test_accs2 = model2_results
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses1) + 1)
    ax1.plot(epochs, train_losses1, 'b-', label=f'{title1} Training Loss')
    ax1.plot(epochs, test_losses1, 'b--', label=f'{title1} Test Loss')
    ax1.plot(epochs, train_losses2, 'r-', label=f'{title2} Training Loss')
    ax1.plot(epochs, test_losses2, 'r--', label=f'{title2} Test Loss')
    ax1.set_title('Loss Comparison')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs1, 'b-', label=f'{title1} Training Accuracy')
    ax2.plot(epochs, test_accs1, 'b--', label=f'{title1} Test Accuracy')
    ax2.plot(epochs, train_accs2, 'r-', label=f'{title2} Training Accuracy')
    ax2.plot(epochs, test_accs2, 'r--', label=f'{title2} Test Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('softmax_comparison.png')
    plt.show()

# 主函数
def main():
    num_epochs = 10
    learning_rate = 0.001
    
    # 模型1：带有内部Softmax的模型
    print("训练带有内部Softmax的模型...")
    model_with_softmax = SoftmaxModelWithActivation()
    # 使用NLLLoss，因为模型已经应用了Softmax
    criterion_with_softmax = nn.NLLLoss()  # 对数似然损失，需要与Softmax配合使用
    optimizer_with_softmax = optim.Adam(model_with_softmax.parameters(), lr=learning_rate)
    
    model1_results = train_model(
        model_with_softmax, trainloader, testloader, 
        criterion_with_softmax, optimizer_with_softmax, 
        "Model with Softmax", num_epochs)
    
    # 模型2：不带Softmax的模型，使用CrossEntropyLoss
    print("\n训练不带Softmax的模型（Softmax在损失函数中）...")
    model_without_softmax = SoftmaxModelWithoutActivation()
    # CrossEntropyLoss内部包含Softmax
    criterion_without_softmax = nn.CrossEntropyLoss()
    optimizer_without_softmax = optim.Adam(model_without_softmax.parameters(), lr=learning_rate)
    
    model2_results = train_model(
        model_without_softmax, trainloader, testloader, 
        criterion_without_softmax, optimizer_without_softmax, 
        "Model without Softmax", num_epochs)
    
    # 绘制对比图
    plot_comparison(
        model1_results, model2_results, 
        "With Softmax in Model", "With Softmax in Loss"
    )
    
    # 打印最终性能
    print("\n最终性能比较:")
    print(f"带Softmax模型 - 测试准确率: {model1_results[3][-1]:.2f}%")
    print(f"不带Softmax模型 - 测试准确率: {model2_results[3][-1]:.2f}%")

if __name__ == "__main__":
    main()
