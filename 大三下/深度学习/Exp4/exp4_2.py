# 导入库保持不变
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 简化数据增广（移除高开销操作）
transform_fast_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # 只保留最有效的增广
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_no_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载优化配置
def get_data_loaders():
    train_dataset = datasets.ImageFolder(
        'D:/review&task/大三下/深度学习/Exp3/hotdog/train', 
        transform=transform_fast_aug if use_augmentation else transform_no_aug
    )
    test_dataset = datasets.ImageFolder(
        'D:/review&task/大三下/深度学习/Exp3/hotdog/test',
        transform=transform_no_aug
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # 增大batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, test_loader

# 模型构建
def build_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model.to(device)

# 训练函数（添加进度显示）
def train_model(model, criterion, optimizer, num_epochs=10):
    from tqdm import tqdm
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段（添加进度条）
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算指标
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    return history

# 主训练流程
def main():
    global train_loader, test_loader, use_augmentation
    
    # 训练有增广模型
    print("\n=== 训练带数据增广的模型 ===")
    use_augmentation = True
    train_loader, test_loader = get_data_loaders()
    
    model = build_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    aug_history = train_model(model, criterion, optimizer, num_epochs=10)
    
    # 训练无增广模型
    print("\n=== 训练无数据增广的模型 ===")
    use_augmentation = False
    train_loader, test_loader = get_data_loaders()
    
    model = build_resnet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    noaug_history = train_model(model, criterion, optimizer, num_epochs=10)
    
    # 结果可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(noaug_history['train_loss'], label='No Aug')
    plt.plot(aug_history['train_loss'], label='With Aug')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(noaug_history['val_acc'], label='No Aug')
    plt.plot(aug_history['val_acc'], label='With Aug')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()