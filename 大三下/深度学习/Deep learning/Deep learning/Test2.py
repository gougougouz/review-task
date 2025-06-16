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

# 定义LeNet模型作为对比
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DummyNet模型
class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        
        # 分支1 - 组件(1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        
        # 分支2 - 组件(2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4),
            nn.LeakyReLU(inplace=True)
        )
        
        # 分支3 - 组件(3)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Sigmoid()
        )
        
        # 添加自适应池化层确保输出尺寸一致
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 组件(5) - 偏卷积1
        self.conv_h = nn.Sequential(
            nn.Conv2d(51, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(inplace=True)
        )
        
        # 组件(7) - 偏卷积2
        self.conv_w = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(inplace=True)
        )
        
        # 组件(9) - 1x1卷积
        self.conv_final = nn.Conv2d(32, 3, kernel_size=1)
        
        # 组件(10) - 全连接层
        self.fc = nn.Linear(3 * 7 * 7, 10)  # 调整为7x7输出
    
    def forward(self, x):
        # 分支1
        branch1_out = self.branch1(x)
        branch1_out = self.adaptive_pool(branch1_out)
        
        # 分支2
        branch2_out = self.branch2(x)
        branch2_out = self.adaptive_pool(branch2_out)
        
        # 分支3
        branch3_out = self.branch3(x)
        branch3_out = self.adaptive_pool(branch3_out)
        
        # 组件(4) - 沿通道维度串接
        concat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        
        # 其余代码保持不变
        conv_h_out = self.conv_h(concat_out)
        mul_out = conv_h_out * branch1_out
        conv_w_out = self.conv_w(mul_out)
        add_out = conv_w_out + mul_out
        conv_final_out = self.conv_final(add_out)
        flatten = conv_final_out.view(conv_final_out.size(0), -1)
        output = self.fc(flatten)
        
        return output

# 训练函数
def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=10):
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
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        test_losses.append(test_loss / len(testloader))
        
        print(f'Epoch {epoch+1} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.2f}%')
    
    return train_losses, test_losses, train_accs, test_accs

# 绘制训练曲线
def plot_curves(train_losses, test_losses, train_accs, test_accs, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title}_curves.png')
    plt.show()

# 主函数
def main():
    num_epochs = 15
    learning_rate = 0.001
    
    # 训练LeNet
    print("训练LeNet模型...")
    lenet = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet.parameters(), lr=learning_rate)
    
    lenet_train_losses, lenet_test_losses, lenet_train_accs, lenet_test_accs = train_model(
        lenet, trainloader, testloader, criterion, optimizer, num_epochs)
    
    # 训练DummyNet
    print("\n训练DummyNet模型...")
    dummynet = DummyNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dummynet.parameters(), lr=learning_rate)
    
    dummy_train_losses, dummy_test_losses, dummy_train_accs, dummy_test_accs = train_model(
        dummynet, trainloader, testloader, criterion, optimizer, num_epochs)
    
    # 绘制训练曲线
    plot_curves(lenet_train_losses, lenet_test_losses, lenet_train_accs, lenet_test_accs, "LeNet")
    plot_curves(dummy_train_losses, dummy_test_losses, dummy_train_accs, dummy_test_accs, "DummyNet")
    
    # 打印最终性能
    print("\n最终性能比较:")
    print(f"LeNet - 测试准确率: {lenet_test_accs[-1]:.2f}%")
    print(f"DummyNet - 测试准确率: {dummy_test_accs[-1]:.2f}%")

if __name__ == "__main__":
    main()
