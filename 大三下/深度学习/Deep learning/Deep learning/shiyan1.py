import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 数据加载
batch_size = 16
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义常量
num_inputs = 28 * 28  # Fashion MNIST图像大小
num_outputs = 10      # 类别数量
num_epochs = 10       # 训练轮数
learning_rate = 0.1   # 学习率

# 定义简单的线性分类器
class SoftmaxRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, X):
        # 注意：不在模型中应用softmax，交叉熵损失函数会处理
        return self.linear(X)

# 定义带有隐藏层的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 计算准确率
def accuracy(y_hat, y):
    """计算预测正确的样本数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 使用argmax获取预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 训练一个epoch
def train_epoch(model, train_loader, loss_fn, optimizer, device='cpu'):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X = X.view(-1, num_inputs)  # 将图像展平
        
        # 前向传播
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    return total_loss / len(train_loader), correct / total

# 评估函数
def evaluate(model, data_loader, loss_fn, device='cpu'):
    """评估模型在给定数据集上的性能"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1, num_inputs)
            
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return total_loss / len(data_loader), correct / total

# 完整训练函数
def train_model(model, train_loader, test_loader, loss_fn, optimizer, num_epochs, device='cpu'):
    """训练模型并记录训练过程"""
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 在测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

# 获取K折交叉验证的数据加载器
def get_data_loaders_k_fold(k, dataset, batch_size):
    """将数据集分割为k折，返回训练和验证数据加载器"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    fold_size = dataset_size // k
    
    splits = []
    for i in range(k):
        val_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = [idx for idx in indices if idx not in val_indices]
        splits.append((train_indices, val_indices))
    
    return splits

# K折交叉验证
def k_fold_cross_validation(k, dataset, test_dataset, batch_size, num_epochs, learning_rate, device='cpu'):
    """执行K折交叉验证"""
    splits = get_data_loaders_k_fold(k, dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f'Fold {fold + 1}/{k}')
        
        # 创建训练和验证数据集
        train_sampler = Subset(dataset, train_idx)
        val_sampler = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sampler, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = SimpleNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        # 训练模型
        fold_results = {'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': []}
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            fold_results['train_losses'].append(train_loss)
            fold_results['train_accs'].append(train_acc)
            
            # 验证
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            fold_results['val_losses'].append(val_loss)
            fold_results['val_accs'].append(val_acc)
            
            print(f'  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 在测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        results.append({
            'Fold': fold + 1,
            'Train Accuracy': train_acc,
            'Validation Accuracy': val_acc,
            'Test Accuracy': test_acc
        })
        
        print(f'Fold {fold+1} - Final Test Accuracy: {test_acc:.4f}')
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# 可视化训练过程
def plot_training_history(history):
    """绘制训练和测试的损失与准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = SimpleNN().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("Starting training...")
    history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
    
    # 可视化训练过程
    plot_training_history(history)
    
    # 执行K折交叉验证
    print("\nStarting K-fold cross validation...")
    k = 5  # 5折交叉验证
    results = k_fold_cross_validation(k, train_dataset, test_dataset, batch_size, num_epochs, learning_rate, device)
    
    # 打印K折交叉验证结果
    print("\nK-fold Cross Validation Results:")
    print(results)
    print(f"Average Test Accuracy: {results['Test Accuracy'].mean():.4f}")

if __name__ == "__main__":
    main()
