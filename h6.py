#!/usr/bin/env python
# coding: utf-8

# ----------
# **类及方法**
# ----------

# In[1]:


# 模块导入
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF


'''
若要禁用打乱标签，将447行ablation注释掉；
使用文件夹读取数据集，请取消85-113行代码注释，并注释掉172-192行数据集类
禁用遮蔽实验时，注释掉375，383行transforms。
'''

# In[2]:


# # 复用用例
# import os
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset, Subset, DataLoader
# from torchvision.transforms import v2 as transforms  # 推荐v2，原生兼容numpy
# def create_big_npy_dataset(
#     data_dir="Dataset",
#     save_img_path="all_images.npy",
#     save_lbl_path="all_labels.npy",
#     target_size=(224, 224)  # 必须和你transform的resize尺寸一致！
# ):

#     # 1. 读取类别和文件路径（复用你原有的逻辑）
#     classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
#     class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
#     images_list = []
#     labels_list = []
    
#     # 2. 遍历所有图片，读取+转RGB+统一尺寸+转numpy
#     for cls_name in classes:
#         cls_dir = os.path.join(data_dir, cls_name)
#         for img_name in os.listdir(cls_dir):
#             if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
#                 img_path = os.path.join(cls_dir, img_name)
#                 # 核心：提前解码+统一尺寸（打包后永久不用再做）
#                 image = Image.open(img_path).convert('RGB').resize(target_size)
#                 images_list.append(np.array(image))  # 转uint8 numpy
#                 labels_list.append(class_to_idx[cls_name])
    
#     # 3. 堆叠成【单个大numpy数组】（速度天花板格式）
#     all_images = np.stack(images_list, axis=0)  # shape: [N, H, W, 3]
#     all_labels = np.array(labels_list)          # shape: [N]
    
#     # 4. 保存为大npy
#     np.save(save_img_path, all_images)
#     np.save(save_lbl_path, all_labels)
    
#     print(f"图片数组形状: {all_images.shape}")
#     print(f"标签数组形状: {all_labels.shape}")
#     print(f"文件已保存: {save_img_path}, {save_lbl_path}")

# # ==============================================
# # create_big_npy_dataset(target_size=(224, 224))


# In[3]:


# # 数据集类
# class MyDataset(Dataset):                              # 自定义数据集类，继承自PyTorch的Dataset基类
#     def __init__(self, data_dir, transform=None):       # 初始化方法，接受数据目录和可选的图像变换
#         self.data_dir = data_dir
#         self.images, self.labels = [], []
#         self.transform = transform
#         classes = sorted([d for d in os.listdir(data_dir) 
#                           if os.path.isdir(os.path.join(data_dir, d))])         # 获取数据目录下的所有子目录（类别），并排序
#         # print(f"Found classes: {classes}")
#         class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}  # 创建类别到索引的映射
#         for cls_name in classes:                                                 # 遍历每个类别目录
#             cls_dir = os.path.join(data_dir, cls_name)                           # 构建类别目录的完整路径
#             for img_name in os.listdir(cls_dir):                                 # 遍历类别目录中的每个文件
#                 if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):          # 仅处理图像文件
#                     self.images.append(os.path.join(cls_dir, img_name))         # 将图像文件的完整路径添加到图像列表中
#                     self.labels.append(class_to_idx[cls_name])                  # 将对应的类别索引添加到标签列表中
#                     # print(f"Added image: {os.path.join(cls_dir, img_name)} with label: {class_to_idx[cls_name]}")
                    

#     def __len__(self):                                                      # 返回数据集的大小，即图像数量
#         return len(self.images)

#     def __getitem__(self, idx):                                            # 根据索引获取图像和标签
#         img_path = self.images[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert('RGB')                         # 打开图像并转换为RGB格式
#         if self.transform:
#             image = self.transform(image)
#         return image, label



# In[4]:
class CircularCrop:
    def __init__(self, radius_ratio=0.8):
        # radius_ratio: 圆形半径占图像短边的比例，0.8可根据数据集调整
        self.radius_ratio = radius_ratio

    def __call__(self, img):
        # img 为 PIL.Image 或 Tensor，这里以Tensor为例
        _, h, w = img.shape
        center_y, center_x = h // 2, w // 2
        radius = int(min(h, w) * self.radius_ratio // 2)

        # 生成圆形掩码
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
        mask = mask.float().unsqueeze(0)  # 对齐通道维度

        # # 掩码相乘，圆外像素置0
        # return img * (mask)
        # 掩码相乘，圆内像素置0
        return img * (1 - mask)
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """
    绘制训练过程中的Loss和Accuracy变化曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history plot saved to {save_path}")


class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # self.images = np.load(img_npy_path)  # [N, H, W, 3]
        # self.labels = np.load(lbl_npy_path)  # [N]
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image) # 转回PIL Image，方便后续transform处理
        label = self.labels[idx]
        
        # 实时数据增强（完全正常工作）
        if self.transform:
            image = self.transform(image)
        
        return image, label


# In[5]:


# 模型

class SimpleCNN(nn.Module):                             # 定义一个简单的卷积神经网络类，继承自PyTorch的nn.Module
    def __init__(self, num_classes=8):                  # 初始化方法，接受可选的类别数
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3x3，填充1
        self.bn1 = nn.BatchNorm2d(16)  # 批量归一化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 第二个卷积层，输入通道数16，输出通道数32，卷积核大小3x3，填充1
        self.bn2 = nn.BatchNorm2d(32)  # 批量归一化层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # 批量归一化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.fc1 = nn.Linear(64*28*28,256)  # 全连接层，输入特征数64*28*28，输出特征数256
        self.fc2 = nn.Linear(256, num_classes)  # 全连接层，输入特征数256，输出特征数为类别数
        self.dropout = nn.Dropout(0.5)  # Dropout层，丢弃率为0.5
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 展平卷积层的输出
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        return x

# 划分数据集
# def split_indices(full_train, train_ratio=0.7, val_ratio=0.15):
#     indices = np.random.permutation(len(full_train))
#     train_end = int(len(full_train) * train_ratio)
#     val_end = int(len(full_train) * (train_ratio + val_ratio))
#     idx_train = indices[:train_end]
#     idx_val = indices[train_end:val_end]
#     idx_test = indices[val_end:]
#     return idx_train, idx_val, idx_test


def split_indices(full_dataset, train_ratio=0.7, val_ratio=0.15, seed=66):
    """
    分层抽样划分数据集索引，确保各类别在训练/验证/测试集中比例一致
    返回:
        idx_train, idx_val, idx_test: 训练/验证/测试集的索引数组
    """
    from collections import defaultdict
    
    # 设置随机种子确保可复现性
    np.random.seed(seed)
    
    labels = np.array(full_dataset.labels)
    unique_classes = np.unique(labels)
    
    idx_train, idx_val, idx_test = [], [], []
    
    # 对每个类别分别进行分层抽样
    for cls in unique_classes:
        # 获取当前类别的所有样本索引
        cls_indices = np.where(labels == cls)[0]
        
        # 打乱当前类别的索引
        np.random.shuffle(cls_indices)
        
        # 计算各类别的划分数量
        n_cls = len(cls_indices)
        n_train = int(n_cls * train_ratio)
        n_val = int(n_cls * val_ratio)
        
        # 划分索引
        idx_train.extend(cls_indices[:n_train].tolist())
        idx_val.extend(cls_indices[n_train:n_train + n_val].tolist())
        idx_test.extend(cls_indices[n_train + n_val:].tolist())
    
    # 转换为numpy数组并再次打乱（避免同类别样本连续）
    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)
    
    # 在每个集合内部再次打乱，避免批次中同类别样本过多
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_val)
    np.random.shuffle(idx_test)
    
    # print(f"数据划分完成:")
    # print(f"  训练集: {len(idx_train)} 样本")
    # print(f"  验证集: {len(idx_val)} 样本")
    # print(f"  测试集: {len(idx_test)} 样本")
    # print(f"  总计: {len(idx_train) + len(idx_val) + len(idx_test)} 样本")
    
    # # 打印各类别分布
    # for cls in unique_classes:
    #     train_count = np.sum(labels[idx_train] == cls)
    #     val_count = np.sum(labels[idx_val] == cls)
    #     test_count = np.sum(labels[idx_test] == cls)
    #     print(f"  类别 {cls}: 训练={train_count}, 验证={val_count}, 测试={test_count}")
    
    return idx_train, idx_val, idx_test


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(loss=running_loss / total, acc=correct / total)
    return running_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    return val_loss / len(val_loader), val_correct / val_total
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
       
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop      



# --------
# **数据增强**
# ----------

# In[ ]:


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                      # 调整图像大小为224x224
    # transforms.CenterCrop((192, 192)),
    transforms.RandomRotation(50),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),                              # 转换为张量
    CircularCrop(radius_ratio=0.7),  # 在转张量后加入
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 标准化图像
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop((192, 192)),
    transforms.ToTensor(),
    CircularCrop(radius_ratio=0.7),  # 在转张量后加入
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = val_transforms



# --------------
# **实例**
# ---------------

# In[ ]:


import torch_directml as dm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import copy

if __name__ == '__main__':
    seed = 101
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = dm.device()  # 获取DirectML设备
    device = 'cpu'
    print(f"Using device: {device}")
    all_images = np.load("all_images.npy")
    all_labels = np.load("all_labels.npy")
    
    full_dataset = MyDataset(
        images=all_images,
        labels=all_labels,
        transform=None  # 变换交给Subset单独设置
    )
    train_idx, val_idx, test_idx = split_indices(full_dataset)
    
    labels_array = np.array(full_dataset.labels)
    class_names = sorted(os.listdir('Dataset'))
    
    print(f"{'类别':<10} {'训练集':<9} {'验证集':<8} {'测试集':<7} {'总计':<6}")
    print("-" * 60)
    
    for cls_idx, cls_name in enumerate(class_names):
        train_count = np.sum(labels_array[train_idx] == cls_idx)
        val_count = np.sum(labels_array[val_idx] == cls_idx)
        test_count = np.sum(labels_array[test_idx] == cls_idx)
        total_count = train_count + val_count + test_count
        print(f"{cls_name:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")
    
    print("-" * 60)
    print(f"{'总计':<12} {len(train_idx):<10} {len(val_idx):<10} {len(test_idx):<10} {len(full_dataset):<10}")
    print("="*60 + "\n")
    shuffled_labels = all_labels.copy()
    np.random.shuffle(shuffled_labels[train_idx])  # 仅打乱训练集的标签
    ablation_train_dataset = Subset(MyDataset(all_images, shuffled_labels, train_transforms), train_idx)
    # train_dataset = Subset(MyDataset("all_images.npy", "all_labels.npy", train_transforms), train_idx)
    # val_dataset   = Subset(MyDataset("all_images.npy", "all_labels.npy", val_transforms), val_idx)
    # test_dataset  = Subset(MyDataset("all_images.npy", "all_labels.npy", test_transforms), test_idx)

    train_dataset = Subset(MyDataset(all_images, all_labels, train_transforms), train_idx)
    train_dataset = ablation_train_dataset  # 使用消融实验数据集
    
    val_dataset   = Subset(MyDataset(all_images, all_labels, val_transforms), val_idx)
    test_dataset  = Subset(MyDataset(all_images, all_labels, test_transforms), test_idx)
    
    

    # # 创建数据集
    # data_dir = 'Dataset'                                  # 数据目录路径
    # train_idx, val_idx, test_idx = split_indices(MyDataset(data_dir), train_ratio=0.7, val_ratio=0.15)  # 获取训练、验证和测试集的索引
    # train_dataset = Subset(MyDataset(data_dir, transform=train_transforms), train_idx)
    # val_dataset   = Subset(MyDataset(data_dir, transform=val_transforms), val_idx)
    # test_dataset  = Subset(MyDataset(data_dir, transform=test_transforms), test_idx)

    batch_size = 77
    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=generator)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=generator)

    model = SimpleCNN(num_classes=8).to(device)  # 创建模型实例，指定类别数为8，并将其移动到指定设备上
    criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 定义优化器为AdamW，设置学习率和权重衰减
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
    # 初始化历史记录列表
    history_train_losses = []
    history_val_losses = []
    history_train_accs = []
    history_val_accs = []
    num_epochs = 100
    best_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_correct = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_correct = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # 记录当前epoch的数据
        history_train_losses.append(train_loss)
        history_val_losses.append(val_loss)
        history_train_accs.append(train_correct)
        history_val_accs.append(val_correct)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_correct:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_correct:.4f}')
        
        if val_correct > best_val_acc:
            best_val_acc = val_correct
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_custom_cnn.pth')
        
        if early_stopping(val_correct):
            print(f'\nEarly stopping triggered at epoch {epoch+1}!')
            print(f'Best validation accuracy: {best_val_acc:.4f}')
            break

    # 训练结束后绘制并保存曲线
    plot_training_history(history_train_losses, history_val_losses, history_train_accs, history_val_accs)






# In[12]:


    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    # 加载最佳模型
    model.load_state_dict(torch.load('best_custom_cnn.pth'))
    Test_loss, test_correct = validate(model, test_loader, criterion, device)
    # 预测
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    # 评估
    y_test = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    class_names = sorted(os.listdir('Dataset'))
    # 分类报告
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(confusion_matrix(y_test, y_pred))
    y_labels = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=y_labels)
    for i, c in enumerate(y_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_val = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {c} (AUC = {roc_auc_val:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{c}.png')
    plt.show()

    cm = confusion_matrix(y_test, y_pred, labels=y_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'C{i}' for i in range(8)], 
                    yticklabels=[f'C{i}' for i in range(8)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{c}.png')
    plt.show()
    print(seed)

