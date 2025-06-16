import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


# 自定义三通道随机噪声
class AddRandomNoise(object):
    def __init__(self, R=0.1):
        self.R = R

    def __call__(self, img):
        noise = torch.FloatTensor(img.size()).uniform_(-self.R, self.R)
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)
        return noisy_img


class CombinedAugmentationDataset(datasets.ImageFolder):
    def __init__(self, root, torch_transform=None, albu_transform=None, target_size=(224, 224)):
        super().__init__(root)
        self.torch_transform = torch_transform
        self.albu_transform = albu_transform
        self.target_size = target_size
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            AddRandomNoise(R=0.05),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)

        if self.torch_transform is not None:
            img = self.torch_transform(img)

        img = img.resize(self.target_size) if isinstance(img, Image.Image) else img
        img_np = np.array(img)

        if self.albu_transform is not None:
            augmented = self.albu_transform(image=img_np)
            img_np = augmented['image']

        img_pil = Image.fromarray(img_np) if img_np.ndim == 3 else Image.fromarray(img_np, mode='L')

        if self.final_transform is not None:
            img = self.final_transform(img_pil)

        return img, target


#  1.1数据准备（有数据增广）
def prepare_data_with_aug(data_dir='D:/review&task/大三下/深度学习/Exp5/hotdog'):
    target_size = (224, 224)

    torchvision_transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    albumentations_transform = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
        ], p=0.2),
    ])

    train_dataset = CombinedAugmentationDataset(
        os.path.join(data_dir, 'train'),
        torch_transform=torchvision_transform,
        albu_transform=albumentations_transform,
        target_size=target_size
    )

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes


# 1.2 数据准备（无数据增广）
def prepare_data_without_aug(data_dir='D:/review&task/大三下/深度学习/Exp5/hotdog'):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes


# 2. 模型构建
def build_model(model_name='resnet18'):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientnet':
        # 使用EfficientNet-B0
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError("Unsupported model name")

    return model


# 3. 测试时增广(TTA)函数
def test_time_augmentation(model, dataloader, num_aug=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 定义TTA变换
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]

    # 限制使用的TTA变换数量
    tta_transforms = tta_transforms[:num_aug]

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 存储所有增强版本的预测
            all_outputs = []

            # 原始图像预测
            outputs = model(inputs)
            all_outputs.append(outputs)

            # 应用TTA变换
            for transform in tta_transforms:
                # 对batch中的每个图像应用变换
                augmented_inputs = torch.stack(
                    [transform(Image.fromarray((x.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))) for x in
                     inputs])
                augmented_inputs = augmented_inputs.to(device)
                outputs = model(augmented_inputs)
                all_outputs.append(outputs)

            # 平均所有预测
            avg_output = torch.mean(torch.stack(all_outputs), dim=0)
            _, predicted = torch.max(avg_output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# 4. 模型集成函数
def ensemble_models(models: List[nn.Module], dataloader, use_tta=False, num_aug=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)
        model.eval()

    if use_tta:
        tta_transforms = [
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
        tta_transforms = tta_transforms[:num_aug]
    else:
        tta_transforms = []

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            all_outputs = []

            for model in models:
                if use_tta:
                    model_outputs = []
                    # 原始图像预测
                    outputs = model(inputs)
                    model_outputs.append(outputs)

                    # TTA预测
                    for transform in tta_transforms:
                        augmented_inputs = torch.stack([
                            transform(Image.fromarray((x.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
                            for x in inputs
                        ])
                        augmented_inputs = augmented_inputs.to(device)
                        outputs = model(augmented_inputs)
                        model_outputs.append(outputs)

                        # 应该在所有TTA变换完成后计算平均
                    avg_output = torch.mean(torch.stack(model_outputs), dim=0)
                    all_outputs.append(avg_output)
                else:
                    outputs = model(inputs)
                    all_outputs.append(outputs)

            # 平均所有模型的预测
            ensemble_output = torch.mean(torch.stack(all_outputs), dim=0)
            _, predicted = torch.max(ensemble_output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
    return accuracy


# 3. 训练和验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, model_name='model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    best_val_acc = 0.0
    best_model_path = f'best_{model_name}.pth'

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n')

        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    return history, model


# 主函数
def main():
    # 参数设置
    data_dir = 'D:/review&task/大三下/深度学习/Exp5/hotdog'
    num_epochs = 20
    lr = 0.001

    # 准备数据
    print("\nPreparing data...")
    train_loader_no_aug, val_loader_no_aug, class_names = prepare_data_without_aug(data_dir)
    train_loader_aug, val_loader_aug, _ = prepare_data_with_aug(data_dir)
    print(f"Class names: {class_names}")

    # 训练ResNet-18无数据增广
    print("\nTraining ResNet-18 without data augmentation...")
    resnet_no_aug = build_model('resnet18')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet_no_aug.parameters(), lr=lr, momentum=0.9)
    history_resnet_no_aug, resnet_no_aug = train_model(resnet_no_aug, train_loader_no_aug, val_loader_no_aug,
                                                       criterion, optimizer, num_epochs, 'resnet18_no_aug')

    # 训练ResNet-18有数据增广
    print("\nTraining ResNet-18 with data augmentation...")
    resnet_aug = build_model('resnet18')
    optimizer = optim.SGD(resnet_aug.parameters(), lr=lr, momentum=0.9)
    history_resnet_aug, resnet_aug = train_model(resnet_aug, train_loader_aug, val_loader_aug,
                                                 criterion, optimizer, num_epochs, 'resnet18_aug')

    # 训练EfficientNet有数据增广
    print("\nTraining EfficientNet with data augmentation...")
    efficientnet_aug = build_model('efficientnet')
    optimizer = optim.SGD(efficientnet_aug.parameters(), lr=lr, momentum=0.9)
    history_efficientnet_aug, efficientnet_aug = train_model(efficientnet_aug, train_loader_aug, val_loader_aug,
                                                             criterion, optimizer, num_epochs, 'efficientnet_aug')

    # 测试准确率
    print("\nEvaluating models...")

    # 基本测试准确率
    resnet_no_aug_acc = ensemble_models([resnet_no_aug], val_loader_no_aug)
    resnet_aug_acc = ensemble_models([resnet_aug], val_loader_aug)
    efficientnet_aug_acc = ensemble_models([efficientnet_aug], val_loader_aug)

    # TTA测试准确率
    resnet_aug_tta_acc = ensemble_models([resnet_aug], val_loader_aug, use_tta=True)
    efficientnet_aug_tta_acc = ensemble_models([efficientnet_aug], val_loader_aug, use_tta=True)

    # 模型集成
    ensemble_acc = ensemble_models([resnet_aug, efficientnet_aug], val_loader_aug, use_tta=True)

    # 打印结果表格
    print("\nResults Table:")
    print("| Model                                   | Train Accuracy | Test Accuracy |")
    print("|-----------------------------------------|----------------|---------------|")
    print(
        f"| ResNet-18                               | {max(history_resnet_no_aug['train_acc']):.4f}         | {resnet_no_aug_acc:.4f}      |")
    print(
        f"| ResNet-18 + DA                          | {max(history_resnet_aug['train_acc']):.4f}         | {resnet_aug_acc:.4f}      |")
    print(
        f"| EfficientNet + DA                       | {max(history_efficientnet_aug['train_acc']):.4f}         | {efficientnet_aug_acc:.4f}      |")
    print(f"| ResNet-18 + DA + TTA                    | N/A            | {resnet_aug_tta_acc:.4f}      |")
    print(f"| EfficientNet + DA + TTA                 | N/A            | {efficientnet_aug_tta_acc:.4f}      |")
    print(f"| ResNet-18 + EfficientNet + DA + TTA     | N/A            | {ensemble_acc:.4f}      |")


if __name__ == '__main__':
    main()