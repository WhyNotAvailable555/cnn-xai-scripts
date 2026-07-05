# -*- coding: utf-8 -*-

"""
Explore feature semantics of SimpleCNN using Sparse Autoencoder (SAE)
SAE can decompose high-dimensional features into interpretable sparse components, helping us understand what the model has learned.
Uses FP16 precision to reduce memory usage.

Note: Import dependencies have been removed from h6.py. All necessary class and function definitions are integrated into this file.
Just make sure to run h6.py first to generate data files (all_images.npy, all_labels.npy) and the model (best_custom_cnn.pth).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import random
GLOBAL_SEED = 42

# ==========================================
# Integrated dependency definitions (from h6.py)
# ==========================================
def set_global_seed(seed=GLOBAL_SEED):
    """
    Set all random seeds for reproducibility
    
    Args:
        seed: Random seed value (default: 66)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed}")

# Set global seed at module level
set_global_seed(GLOBAL_SEED)

# ----------
# Dataset Class
# ----------
class MyDataset(Dataset):
    """Custom dataset class that loads data from numpy arrays"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)  # Convert back to PIL Image for subsequent transform processing
        label = self.labels[idx]
        
        # Real-time data augmentation
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ----------
# CNN Model
# ----------
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


# ----------
# Data Splitting Function
# ----------
# def split_indices(full_train, train_ratio=0.7, val_ratio=0.15):
#     """Split dataset indices into training, validation, and test sets"""
#     indices = np.random.permutation(len(full_train))
#     train_end = int(len(full_train) * train_ratio)
#     val_end = int(len(full_train) * (train_ratio + val_ratio))
#     idx_train = indices[:train_end]
#     idx_val = indices[train_end:val_end]
#     idx_test = indices[val_end:]
#     return idx_train, idx_val, idx_test
# ... existing code ...

# ... existing code ...

def split_indices(full_dataset, train_ratio=0.7, val_ratio=0.15, seed=None):
    """
    分层抽样划分数据集索引，确保各类别在训练/验证/测试集中比例一致
    
    Args:
        full_dataset: 完整数据集对象，需包含labels属性
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.15
        seed: 随机种子，None则使用全局种子
    
    Returns:
        idx_train, idx_val, idx_test: 训练/验证/测试集的索引数组
    """
    from collections import defaultdict
    
    # Use provided seed or global seed
    if seed is None:
        seed = GLOBAL_SEED
    
    # Create local RNG instance to avoid affecting global state
    rng = np.random.RandomState(seed)
    
    labels = np.array(full_dataset.labels)
    unique_classes = np.unique(labels)
    
    idx_train, idx_val, idx_test = [], [], []
    
    # 对每个类别分别进行分层抽样
    for cls in unique_classes:
        # 获取当前类别的所有样本索引
        cls_indices = np.where(labels == cls)[0]
        
        # 打乱当前类别的索引
        rng.shuffle(cls_indices)
        
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
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)
    
    print(f"数据划分完成:")
    print(f"  训练集: {len(idx_train)} 样本")
    print(f"  验证集: {len(idx_val)} 样本")
    print(f"  测试集: {len(idx_test)} 样本")
    print(f"  总计: {len(idx_train) + len(idx_val) + len(idx_test)} 样本")
    
    # 打印各类别分布
    for cls in unique_classes:
        train_count = np.sum(labels[idx_train] == cls)
        val_count = np.sum(labels[idx_val] == cls)
        test_count = np.sum(labels[idx_test] == cls)
        print(f"  类别 {cls}: 训练={train_count}, 验证={val_count}, 测试={test_count}")
    
    return idx_train, idx_val, idx_test

# ... existing code ...
# # ... existing code ...
# class CircularCrop:
#     def __init__(self, radius_ratio=0.8):
#         # radius_ratio: 圆形半径占图像短边的比例，0.8可根据数据集调整
#         self.radius_ratio = radius_ratio

#     def __call__(self, img):
#         # img 为 PIL.Image 或 Tensor，这里以Tensor为例
#         _, h, w = img.shape
#         center_y, center_x = h // 2, w // 2
#         radius = int(min(h, w) * self.radius_ratio // 2)

#         # 生成圆形掩码
#         y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
#         mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
#         mask = mask.float().unsqueeze(0)  # 对齐通道维度

#         # 掩码相乘，圆外像素置0
#         return img * mask
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
        return img * (mask)
        # 掩码相乘，圆内像素置0
        # return img * (1 - mask)

# ----------
# Data Transforms
# ----------
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop((192, 192)),
    transforms.ToTensor(),
    # CircularCrop(radius_ratio=1),  # 在转张量后加入
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# End of integrated dependency definitions
# ==========================================

# --------------------------
# 1. Sparse Autoencoder (SAE) Implementation
# --------------------------
# class SparseAutoencoder(nn.Module):
#     """
#     Sparse Autoencoder: Encodes input features into sparse representations, then decodes for reconstruction
#     Enforces sparsity through L1 regularization, making each neuron correspond to specific semantic concepts
#     Uses FP16 precision to reduce memory usage
#     """
#     def __init__(self, input_dim, hidden_dim, sparsity_weight=0.01):
#         super(SparseAutoencoder, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.sparsity_weight = sparsity_weight
        
#         # Encoder: input_dim -> hidden_dim
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # Decoder: hidden_dim -> input_dim
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, input_dim)
#             # Remove Sigmoid, allowing the model to learn feature ranges freely
#         )
        
#         # Remove FP16 conversion for better numerical stability
#         # self.half()  # Commented out to use FP32
    
#     def forward(self, x):
#         # Use FP32 for better numerical stability
#         # if x.dtype != torch.float16:
#         #     x = x.half()
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
    
#     # def sparse_loss(self, encoded, target_sparsity=0.05):
#         """
#         Calculate sparsity loss (KL divergence)
#         Encourages average neuron activation to approach target_sparsity
#         """
#         # Calculate average activation for each neuron
#         avg_activation = torch.mean(encoded, dim=0)
        
#         # Clip to safe range to avoid log(0) and numerical issues
#         # Use wider clipping range for better stability
#         avg_activation = torch.clamp(avg_activation, min=1e-6, max=0.999999)
        
#         # KL divergence: measures difference between actual distribution and target sparse distribution
#         # Add small epsilon to prevent division by zero
#         eps = 1e-8
#         kl_divergence = target_sparsity * torch.log((target_sparsity + eps) / (avg_activation + eps)) + \
#                        (1 - target_sparsity) * torch.log((1 - target_sparsity + eps) / (1 - avg_activation + eps))
        
#         # Clamp KL divergence to prevent explosion
#         kl_divergence = torch.clamp(kl_divergence, min=-10, max=10)
        
#         return torch.sum(kl_divergence)
#     def sparse_loss(self, encoded, target_sparsity=0.05):
#         """
#         Calculate sparsity loss (KL divergence)
#         Encourages average neuron activation to approach target_sparsity
#         """
#         # Calculate average activation for each neuron
#         avg_activation = torch.mean(encoded, dim=0)
        
#         # Clip to safe range to avoid log(0) and numerical issues
#         avg_activation = torch.clamp(avg_activation, min=1e-8, max=1.0-1e-8)
        
#         # KL divergence: measures difference between actual distribution and target sparse distribution
#         eps = 1e-8
#         kl_per_neuron = target_sparsity * torch.log((target_sparsity + eps) / (avg_activation + eps)) + \
#                        (1 - target_sparsity) * torch.log((1 - target_sparsity + eps) / (1 - avg_activation + eps))
        
#         # Clamp individual neuron KL to prevent extreme values
#         kl_per_neuron = torch.clamp(kl_per_neuron, min=0.0, max=50.0)
        
#         # Return mean instead of sum to keep loss scale reasonable
#         return torch.mean(kl_per_neuron)
# ... existing code ...
class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder: Encodes input features into sparse representations, then decodes for reconstruction
    Uses Top-K hard sparsity instead of KL divergence soft sparsity
    
    Args:
        k: Number of top activations to keep (for hard sparsity)
        use_hard_sparsity: Whether to use Top-K hard sparsity (default: True)
    """
    def __init__(self, input_dim, hidden_dim, sparsity_weight=0.01, k=None, use_hard_sparsity=True):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.k = k if k is not None else max(1, hidden_dim // 20)  # Default: keep top 5%
        self.use_hard_sparsity = use_hard_sparsity
        
        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: hidden_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )
    
    def apply_topk_sparsity(self, encoded):
        """
        Apply Top-K hard sparsity: keep only the top-k activations, zero out others
        
        Args:
            encoded: [batch_size, hidden_dim]
        Returns:
            sparse_encoded: [batch_size, hidden_dim] with only top-k values non-zero
        """
        if not self.use_hard_sparsity:
            return encoded
        
        batch_size = encoded.shape[0]
        
        # Find top-k values and their indices for each sample in the batch
        top_k_values, top_k_indices = torch.topk(encoded, self.k, dim=1)
        
        # Create a sparse tensor with zeros
        sparse_encoded = torch.zeros_like(encoded)
        
        # Scatter the top-k values back to their original positions
        sparse_encoded.scatter_(1, top_k_indices, top_k_values)
        
        return sparse_encoded
    
    def forward(self, x):
        """
        Forward pass with optional Top-K hard sparsity
        """
        encoded = self.encoder(x)
        
        # Apply Top-K hard sparsity after encoder
        if self.use_hard_sparsity:
            encoded_sparse = self.apply_topk_sparsity(encoded)
        else:
            encoded_sparse = encoded
        
        decoded = self.decoder(encoded_sparse)
        
        # Return both raw and sparse encoded for flexibility
        return encoded_sparse, decoded
    
    def sparse_loss(self, encoded, target_sparsity=0.05):
        """
        For Top-K hard sparsity, this loss is not needed but kept for compatibility.
        Returns 0 when using hard sparsity.
        """
        if self.use_hard_sparsity:
            return torch.tensor(0.0, device=encoded.device)
        
        # Fallback to KL divergence if not using hard sparsity
        avg_activation = torch.mean(encoded, dim=0)
        avg_activation = torch.clamp(avg_activation, min=1e-8, max=1.0-1e-8)
        
        eps = 1e-8
        kl_per_neuron = target_sparsity * torch.log((target_sparsity + eps) / (avg_activation + eps)) + \
                       (1 - target_sparsity) * torch.log((1 - target_sparsity + eps) / (1 - avg_activation + eps))
        
        kl_per_neuron = torch.clamp(kl_per_neuron, min=0.0, max=50.0)
        return torch.mean(kl_per_neuron)
# ... existing code ...
# --------------------------
# 2. SAE Training Function
# --------------------------
# def train_sae(sae, features, num_epochs=100, lr=0.001, batch_size=256, device='cpu'):
#     """
#     Train sparse autoencoder (using FP16 precision)
#     """
#     sae = sae.to(device)
#     optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
#     # Convert features to FP32 for better numerical stability
#     features_tensor = torch.FloatTensor(features).to(device)
#     dataset = torch.utils.data.TensorDataset(features_tensor)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     losses = []
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         epoch_recon_loss = 0
#         epoch_sparse_loss = 0
#         num_batches = 0
        
#         for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
#             x = batch[0]
            
#             # Forward propagation
#             encoded, decoded = sae(x)
            
#             # Reconstruction loss (MSE)
#             recon_loss = F.mse_loss(decoded, x)
            
#             # Sparsity loss with better numerical stability
#             sparse_loss_val = sae.sparse_loss(encoded, target_sparsity=0.05)
            
#             # Check for NaN and skip problematic batches
#             if torch.isnan(recon_loss) or torch.isnan(sparse_loss_val) or torch.isinf(recon_loss) or torch.isinf(sparse_loss_val):
#                 print(f"Warning: NaN/Inf detected at epoch {epoch+1}, skipping batch")
#                 continue
            
#             # Total loss
#             total_loss = recon_loss + sae.sparsity_weight * sparse_loss_val
            
#             # Check total loss before backward
#             if torch.isnan(total_loss) or torch.isinf(total_loss):
#                 print(f"Warning: NaN/Inf in total loss at epoch {epoch+1}, skipping batch")
#                 continue
            
#             # Backpropagation
#             optimizer.zero_grad()
#             total_loss.backward()
            
#             # Gradient clipping to prevent explosion
#             torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             epoch_loss += total_loss.item()
#             epoch_recon_loss += recon_loss.item()
#             epoch_sparse_loss += sparse_loss_val.item()
#             num_batches += 1
        
#         if num_batches == 0:
#             print(f"Warning: All batches skipped in epoch {epoch+1}")
#             continue
        
#         avg_loss = epoch_loss / num_batches
#         avg_recon = epoch_recon_loss / num_batches
#         avg_sparse = epoch_sparse_loss / num_batches
#         losses.append(avg_loss)
        
#         if (epoch + 1) % 1 == 0:
#             print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Sparse={avg_sparse:.4f}')
    
#     return sae, losses
# ... existing code ...

# def train_sae(sae, features, num_epochs=100, lr=0.001, 
#                                    batch_size=256, device='cpu',
#                                    patience=10, min_delta=1e-4, seed=None):
#     """
#     训练稀疏自编码器，支持早停
    
#     Args:
#         patience: 容忍多少个 epoch 没有改善就停止
#         min_delta: 认为损失改善的最小阈值
#         seed: 随机种子用于DataLoader shuffle
#     """
#     sae = sae.to(device)
#     optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
#     # Use provided seed or global seed
#     if seed is None:
#         seed = GLOBAL_SEED
    
#     features_tensor = torch.FloatTensor(features).to(device)
#     dataset = torch.utils.data.TensorDataset(features_tensor)
    
#     # Create generator with fixed seed for reproducibility
#     generator = torch.Generator()
#     generator.manual_seed(seed)
    
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
#     losses = []
#     best_loss = float('inf')
#     patience_counter = 0
#     best_state_dict = None
    
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         epoch_recon_loss = 0
#         epoch_sparse_loss = 0
#         num_batches = 0
        
#         for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
#             x = batch[0]
#             encoded, decoded = sae(x)
            
#             recon_loss = F.mse_loss(decoded, x)
#             sparse_loss_val = sae.sparse_loss(encoded, target_sparsity=0.05)
            
#             if torch.isnan(recon_loss) or torch.isnan(sparse_loss_val):
#                 continue
            
#             total_loss = recon_loss + sae.sparsity_weight * sparse_loss_val
            
#             if torch.isnan(total_loss):
#                 continue
            
#             optimizer.zero_grad()
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             epoch_loss += total_loss.item()
#             epoch_recon_loss += recon_loss.item()
#             epoch_sparse_loss += sparse_loss_val.item()
#             num_batches += 1
        
#         if num_batches == 0:
#             continue
        
#         avg_loss = epoch_loss / num_batches
#         avg_recon = epoch_recon_loss / num_batches
#         avg_sparse = epoch_sparse_loss / num_batches
#         losses.append(avg_loss)
        
#         if (epoch + 1) % 1 == 0:
#             print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Sparse={avg_sparse:.4f}')
        
#         # 早停检查
#         if avg_loss < best_loss - min_delta:
#             best_loss = avg_loss
#             patience_counter = 0
#             best_state_dict = sae.state_dict().copy()  # 保存最佳模型
#             print(f"  ✓ New best model saved (loss: {best_loss:.4f})")
#         else:
#             patience_counter += 1
#             print(f"  ⏸ No improvement ({patience_counter}/{patience})")
        
#         if patience_counter >= patience:
#             print(f"\n🛑 Early stopping at epoch {epoch+1}")
#             print(f"Best loss: {best_loss:.4f}")
#             # 恢复最佳模型
#             if best_state_dict is not None:
#                 sae.load_state_dict(best_state_dict)
#             break
    
#     return sae, losses
# ... existing code ...
def train_sae(sae, features, num_epochs=100, lr=0.001, 
                                   batch_size=256, device='cpu',
                                   patience=5, min_delta=1e-4, seed=None):
    """
    训练稀疏自编码器，支持早停和Top-K硬稀疏
    
    Args:
        patience: 容忍多少个 epoch 没有改善就停止
        min_delta: 认为损失改善的最小阈值
        seed: 随机种子用于DataLoader shuffle
    """
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Use provided seed or global seed
    if seed is None:
        seed = GLOBAL_SEED
    
    features_tensor = torch.FloatTensor(features).to(device)
    dataset = torch.utils.data.TensorDataset(features_tensor)
    
    # Create generator with fixed seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_sparse_loss = 0
        num_batches = 0
        
        # For tracking dead neurons
        total_activations = torch.zeros(sae.hidden_dim, device=device)
        
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            x = batch[0]
            encoded, decoded = sae(x)
            
            recon_loss = F.mse_loss(decoded, x)
            
            # For hard sparsity, sparse_loss returns 0
            sparse_loss_val = sae.sparse_loss(encoded, target_sparsity=0.05)
            
            if torch.isnan(recon_loss) or torch.isnan(sparse_loss_val):
                continue
            
            # When using hard sparsity, only use reconstruction loss
            total_loss = recon_loss + sae.sparsity_weight * sparse_loss_val
            
            if torch.isnan(total_loss):
                continue
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_sparse_loss += sparse_loss_val.item()
            num_batches += 1
            
            # Accumulate activations for dead neuron detection
            # Use the sparse encoded output (after Top-K)
            total_activations += encoded.sum(dim=0)
        
        if num_batches == 0:
            continue
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_sparse = epoch_sparse_loss / num_batches
        losses.append(avg_loss)
        
        # Calculate dead neuron ratio
        # A neuron is "dead" if it never activated (sum of activations == 0) across all batches
        dead_neurons = (total_activations == 0).sum().item()
        dead_neuron_ratio = dead_neurons / sae.hidden_dim * 100
        
        if (epoch + 1) % 1 == 0:
            sparsity_type = "Hard (Top-K)" if sae.use_hard_sparsity else "Soft (KL)"
            print(f'Epoch {epoch+1} [{sparsity_type}]: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Sparse={avg_sparse:.4f}')
            print(f'  Dead neurons: {dead_neurons}/{sae.hidden_dim} ({dead_neuron_ratio:.2f}%)')
        
        # 早停检查
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_state_dict = sae.state_dict().copy()
            print(f"  ✓ New best model saved (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏸ No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"Best loss: {best_loss:.4f}")
            print(f"Final dead neuron ratio: {dead_neuron_ratio:.2f}%")
            if best_state_dict is not None:
                sae.load_state_dict(best_state_dict)
            break
    
    return sae, losses
# ... existing code ...
# ... existing code ...
# --------------------------
# 3. Feature Extraction Function
# --------------------------
def extract_conv_features(model, loader, device, extra_pooling=False, normalize=True):
    """
    Extract features after the conv3 layer (before entering fully connected layers)
    
    Args:
        model: The CNN model
        loader: Data loader
        device: Device to run on
        extra_pooling: If True, apply additional pooling to reduce spatial dimensions from 28x28 to 7x7
        normalize: If True, normalize features per-sample to prevent channel dominance
    """
    model.eval()
    features = []
    labels = []
    image_paths = []
    
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            
            # Forward propagation to after conv3 pool
            x = model.pool(F.relu(model.conv1(images)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            
            # Apply extra pooling if requested (28x28 -> 14x14 -> 7x7)
            if extra_pooling:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 28x28 -> 14x14
                x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 14x14 -> 7x7
            
            # Flatten features: [batch, channels, H, W] -> [batch, channels*H*W]
            x_flat = x.view(x.size(0), -1)
            
            # Normalize features per sample to prevent channel dominance
            if normalize:
                # Min-Max normalization to [0, 1] per sample
                min_val = x_flat.min(dim=1, keepdim=True)[0]
                max_val = x_flat.max(dim=1, keepdim=True)[0]
                range_val = max_val - min_val + 1e-8
                x_flat = (x_flat - min_val) / range_val
            
            features.append(x_flat.cpu().numpy())
            labels.append(lbls.numpy())
    
    return np.vstack(features), np.concatenate(labels)


# --------------------------
# 4. Visualize Dictionary Atoms Learned by SAE
# --------------------------
def visualize_dictionary(sae, input_shape, class_names, num_atoms=20, device='cpu'):
    """
    Visualize sparse dictionary atoms learned by SAE
    Each atom represents a learned semantic concept
    """
    sae.eval()
    
    # Get decoder weights (dictionary atoms) - Convert to FP32 for visualization
    decoder_weights = sae.decoder[0].weight.cpu().detach().float().numpy()  # [hidden_dim, input_dim]
    
    # Select the most active num_atoms atoms
    atom_norms = np.linalg.norm(decoder_weights, axis=1)
    top_atoms_idx = np.argsort(atom_norms)[-num_atoms:]
    
    # Visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, atom_idx in enumerate(top_atoms_idx):
        atom = decoder_weights[atom_idx]
        
        # Reshape into spatial structure (assuming input is 64*28*28)
        # Here we simply show the distribution of weights
        axes[i].hist(atom, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_title(f'Atom {atom_idx}\nNorm: {atom_norms[atom_idx]:.2f}', fontsize=9)
        axes[i].set_xlabel('Weight Value')
        axes[i].set_ylabel('Frequency')
    
    plt.suptitle('SAE Dictionary Atoms Distribution', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('sae_dictionary_atoms.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Dictionary atoms visualization saved as 'sae_dictionary_atoms.png'")


# --------------------------
# 5. Analyze Sparse Activation Patterns of Features
# --------------------------
def analyze_sparse_activations(sae, features, labels, class_names, device='cpu'):
    """
    Analyze activation patterns of different category samples in SAE hidden layer
    """
    sae.eval()
    # Use FP32 for consistency with model parameters
    features_tensor = torch.FloatTensor(features).to(device)
    
    with torch.no_grad():
        encoded, _ = sae(features_tensor)
        # No need to convert, already in FP32
        encoded_np = encoded.cpu().numpy()
    
    # Calculate average activation for each category
    num_classes = len(class_names)
    hidden_dim = encoded_np.shape[1]
    class_avg_activations = np.zeros((num_classes, hidden_dim))
    
    for c in range(num_classes):
        mask = labels == c
        if np.sum(mask) > 0:
            class_avg_activations[c] = np.mean(encoded_np[mask], axis=0)
    
    # Find the most discriminative atoms for each category
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for c in range(min(num_classes, 8)):
        # Find top-10 atoms with strongest activation for this category
        top_atoms = np.argsort(class_avg_activations[c])[-10:]
        
        axes[c].bar(range(10), class_avg_activations[c][top_atoms], color='steelblue')
        axes[c].set_title(f'Class: {class_names[c]}', fontsize=11)
        axes[c].set_xlabel('Top Activated Atoms')
        axes[c].set_ylabel('Activation Strength')
        axes[c].set_xticks(range(10))
        axes[c].set_xticklabels([str(a) for a in top_atoms], rotation=45, ha='right', fontsize=7)
    
    plt.suptitle('Class-Specific SAE Atom Activations', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('sae_class_activations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Class-specific activations saved as 'sae_class_activations.png'")
    
    return class_avg_activations


# --------------------------
# 6. Reconstruction Quality Analysis
# --------------------------
# ... existing code ...

def analyze_reconstruction_quality(sae, features, sample_indices, device='cpu'):
    """
    Analyze SAE reconstruction quality, comparing original and reconstructed features
    """
    sae.eval()
    # Use FP32 for consistency with model parameters
    features_tensor = torch.FloatTensor(features[sample_indices]).to(device)
    
    with torch.no_grad():
        encoded, decoded = sae(features_tensor)
    
    # No conversion needed, already in FP32
    original = features_tensor.cpu().numpy()
    reconstructed = decoded.cpu().numpy()
    
    # Calculate reconstruction error
    mse_per_sample = np.mean((original - reconstructed) ** 2, axis=1)
    
    # Visualize reconstruction comparison for several samples
    fig, axes = plt.subplots(3, len(sample_indices), figsize=(4*len(sample_indices), 12))
    
    for i, idx in enumerate(sample_indices):
        # Original features
        axes[0, i].plot(original[i][:100], 'b-', linewidth=0.5, label='Original')
        axes[0, i].set_title(f'Sample {idx}\nMSE: {mse_per_sample[i]:.4f}')
        axes[0, i].legend(fontsize=8)
        axes[0, i].set_ylabel('Feature Value')
        
        # Reconstructed features
        axes[1, i].plot(reconstructed[i][:100], 'r-', linewidth=0.5, label='Reconstructed')
        axes[1, i].legend(fontsize=8)
        axes[1, i].set_ylabel('Feature Value')
        
        # Error
        axes[2, i].plot(np.abs(original[i][:100] - reconstructed[i][:100]), 'g-', linewidth=0.5)
        axes[2, i].set_xlabel('Feature Dimension')
        axes[2, i].set_ylabel('Absolute Error')
    
    plt.suptitle('SAE Reconstruction Quality Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('sae_reconstruction_quality.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Reconstruction quality saved as 'sae_reconstruction_quality.png'")
    
    return mse_per_sample

# ... existing code ...# 8. Explore Specific Neuron Semantics
# --------------------------
def explore_neuron_semantics(sae, model, loader, device, neuron_idx, class_names, top_k=9):
    """
    Explore the semantics of a specific SAE neuron by finding maximally activating images.
    """
    sae.eval()
    model.eval()
    
    all_activations = []
    all_images = []
    all_labels = []
    
    print(f"Extracting activations for neuron {neuron_idx}...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Scanning dataset"):
            images = images.to(device)
            
            # 1. Extract CNN features (same as extract_conv_features logic)
            x = model.pool(F.relu(model.conv1(images)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            
            # Apply extra pooling if your SAE was trained on pooled features
            # Note: Ensure this matches the 'use_extra_pooling' setting in your main block
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            
            x_flat = x.view(x.size(0), -1)
            
            # 2. Get SAE activations
            encoded, _ = sae(x_flat)
            
            # Store activation of the specific neuron
            all_activations.append(encoded[:, neuron_idx].cpu().numpy())
            all_images.append(images.cpu())
            all_labels.append(labels.numpy())
    
    all_activations = np.concatenate(all_activations)
    all_images = torch.cat(all_images, dim=0)
    all_labels = np.concatenate(all_labels)
    
    # 3. Find Top-K maximally activating indices
    top_indices = np.argsort(all_activations)[-top_k:][::-1]
    
    # 4. Visualize
    fig, axes = plt.subplots(1, top_k, figsize=(3*top_k, 3))
    if top_k == 1:
        axes = [axes]
        
    for i, idx in enumerate(top_indices):
        img = all_images[idx]
        # Unnormalize for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img * std + mean
        img_display = torch.clamp(img_display, 0, 1).permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_display)
        true_label = class_names[all_labels[idx]]
        act_val = all_activations[idx]
        axes[i].set_title(f"{true_label}\nAct: {act_val:.2f}", fontsize=8)
        axes[i].axis('off')
        
    plt.suptitle(f'Top {top_k} Images Activating Neuron {neuron_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'sae_neuron_{neuron_idx}_activations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved as 'sae_neuron_{neuron_idx}_activations.png'")
    
    # 5. Visualize the Decoder Weight for this neuron (if input dim allows reshaping)
    # This shows what feature pattern the neuron "looks for" in the flattened space
    decoder_weights = sae.decoder[0].weight.cpu().detach().numpy() # [hidden_dim, input_dim]
    print('750行左右，explore_neuron_semantics函数中，神经元权重有问题，需检查...')
    # neuron_weight = decoder_weights[:,neuron_idx] # [input_dim]
    neuron_weight = decoder_weights[:,neuron_idx] # [input_dim]
    # Try to reshape back to spatial dimensions (assuming 64 channels)
    # Input dim should be 64 * H * W
    input_dim = sae.input_dim
    channels = 64
    if input_dim % channels == 0:
        spatial_size = int(np.sqrt(input_dim // channels))
        if spatial_size * spatial_size * channels == input_dim:
            weight_map = neuron_weight.reshape(channels, spatial_size, spatial_size)
            # Average across channels to get a 2D heatmap of where it attends
            attention_map = np.max(np.abs(weight_map), axis=0)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(attention_map, cmap='hot')
            plt.title(f'Neuron {neuron_idx} Avg Attention Map ({spatial_size}x{spatial_size})')
            plt.colorbar()
            plt.savefig(f'sae_neuron_{neuron_idx}_attention.png', dpi=300)
            plt.show()
            print(f"Attention map saved as 'sae_neuron_{neuron_idx}_attention.png'")


# --------------------------
# 7. Main Program
# --------------------------
# ... existing code ...

# --------------------------
# 7. Main Program
# --------------------------
if __name__ == '__main__':
    # Set random seed (already set at module level, but explicit here for clarity)
    set_global_seed(GLOBAL_SEED)
    
    # Set device
    device = torch.device('cpu')
    try:
        import torch_directml as dm
        device = dm.device()
        device = 'cpu'  # Force CPU for better numerical stability in SAE training
        print(f"Using DirectML device: {device}")
    except:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {device}")
        else:
            print("Using CPU")
    
    # Load data
    if not os.path.exists("all_images.npy") or not os.path.exists("all_labels.npy"):
        raise FileNotFoundError("Please run h6.py first to generate all_images.npy and all_labels.npy")
    
    print("Loading data...")
    all_images = np.load("all_images.npy")
    all_labels = np.load("all_labels.npy")
    full_dataset = MyDataset(images=all_images, labels=all_labels, transform=val_transforms)
    
    # Use the same split as during training
    train_idx, val_idx, test_idx = split_indices(full_dataset, seed=GLOBAL_SEED)
    
    # Create test set loader
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    class_names = sorted([d for d in os.listdir('Dataset') if os.path.isdir(os.path.join('Dataset', d))])
    print(f"Classes: {class_names}")
    
    # Load pretrained model
    if not os.path.exists('best_custom_cnn.pth'):
        raise FileNotFoundError("Please run h6.py first to train the model")
    
    print("Loading pretrained model...")
    model = SimpleCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load('best_custom_cnn.pth', map_location=device))
    model.eval()
    print("Model loaded successfully.")
    

    # ==========================================
    # Step 1: Extract Features
    # ==========================================
    print("\n" + "="*60)
    print("Step 1: Extracting features from conv3 layer...")
    print("="*60)
    
    # Set to True to apply extra pooling (28x28 -> 7x7)
    # This reduces feature dimension from 50176 to 3136, making SAE training more efficient
    use_extra_pooling = True
    # use_extra_pooling = False
    
    # Enable normalization to prevent large-value channels from dominating training
    normalize_features = True
    
    features, labels = extract_conv_features(model, test_loader, device, 
                                            extra_pooling=use_extra_pooling,
                                            normalize=normalize_features)
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Feature statistics:")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    
    if use_extra_pooling:
        print("Note: Extra pooling applied - features reduced from 64*28*28 to 64*7*7")
    else:
        print("Note: Using original feature size 64*28*28")
    
    if normalize_features:
        print("Note: Features normalized (Min-Max to [0,1]) to prevent channel dominance")
    
# ... existing code ...
    
    # # ==========================================
    # # Step 2: Train SAE
    # # ==========================================
    # print("\n" + "="*60)
    # print("Step 2: Training Sparse Autoencoder...")
    # print("="*60)
    
    # input_dim = features.shape[1]  # Feature dimension
    # hidden_dim = 64*7*7*4  # SAE hidden layer dimension (dictionary size)
    # sparsity_weight = 0.01  # Reduce sparsity weight to avoid numerical instability
    
    # print(f"Input dimension: {input_dim}")
    # print(f"Hidden dimension (dictionary size): {hidden_dim}")
    # print(f"Sparsity weight: {sparsity_weight}")
    
    # # Memory usage estimation (using FP32 for stability)
    # num_params_encoder = input_dim * hidden_dim + hidden_dim
    # num_params_decoder = hidden_dim * input_dim + input_dim
    # total_params = num_params_encoder + num_params_decoder
    
    # param_memory_gb = (total_params * 4) / (1024**3)  # FP32: 4 bytes/param
    # optimizer_memory_gb = (total_params * 4 * 2) / (1024**3)  # Adam: 2 states × 4 bytes
    # activation_memory_mb = 256 * hidden_dim * 4 / (1024**2)  # batch_size=256, FP32
    
    # print(f"\n Memory Estimation (FP32 for stability):")
    # print(f"   Total parameters: {total_params:,}")
    # print(f"   Model parameters: ~{param_memory_gb:.2f} GB")
    # print(f"   Optimizer states: ~{optimizer_memory_gb:.2f} GB")
    # print(f"   Activations (per batch): ~{activation_memory_mb:.2f} MB")
    # print(f"   Estimated total: ~{param_memory_gb + optimizer_memory_gb:.2f} GB\n")
    
    # sae = SparseAutoencoder(input_dim=input_dim, 
    #                        hidden_dim=hidden_dim, 
    #                        sparsity_weight=sparsity_weight)
    
    # sae, training_losses = train_sae(sae, features, 
    #                                  num_epochs=100, 
    #                                  lr=0.0005,  # Reduced learning rate for stability
    #                                  batch_size=32,  # Increased batch size
    #                                  device=device,
    #                                  seed=GLOBAL_SEED)
    # ... existing code ...
    # ==========================================
    # Step 2: Train SAE
    # ==========================================
    print("\n" + "="*60)
    print("Step 2: Training Sparse Autoencoder...")
    print("="*60)
    
    input_dim = features.shape[1]  # Feature dimension
    hidden_dim = 64*7*7*8  # SAE hidden layer dimension (dictionary size)
    sparsity_weight = 0.01  # Reduce sparsity weight to avoid numerical instability
    
    # Configure Top-K hard sparsity
    use_hard_sparsity = True  # Set to True for Top-K, False for KL divergence
    k_value = 64  # Keep top 5% of neurons (adjust as needed)
    
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension (dictionary size): {hidden_dim}")
    print(f"Sparsity method: {'Top-K Hard' if use_hard_sparsity else 'KL Soft'}")
    if use_hard_sparsity:
        print(f"Top-K value: {k_value} (keeping top {k_value/hidden_dim*100:.1f}% activations)")
    else:
        print(f"Sparsity weight: {sparsity_weight}")
    
    # Memory usage estimation (using FP32 for stability)
    num_params_encoder = input_dim * hidden_dim + hidden_dim
    num_params_decoder = hidden_dim * input_dim + input_dim
    total_params = num_params_encoder + num_params_decoder
    
    param_memory_gb = (total_params * 4) / (1024**3)  # FP32: 4 bytes/param
    optimizer_memory_gb = (total_params * 4 * 2) / (1024**3)  # Adam: 2 states × 4 bytes
    activation_memory_mb = 256 * hidden_dim * 4 / (1024**2)  # batch_size=256, FP32
    
    print(f"\n Memory Estimation (FP32 for stability):")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model parameters: ~{param_memory_gb:.2f} GB")
    print(f"   Optimizer states: ~{optimizer_memory_gb:.2f} GB")
    print(f"   Activations (per batch): ~{activation_memory_mb:.2f} MB")
    print(f"   Estimated total: ~{param_memory_gb + optimizer_memory_gb:.2f} GB\n")
    
    # Initialize SAE with Top-K hard sparsity
    sae = SparseAutoencoder(input_dim=input_dim, 
                           hidden_dim=hidden_dim, 
                           sparsity_weight=sparsity_weight,
                           k=k_value,
                           use_hard_sparsity=use_hard_sparsity)
    
    sae, training_losses = train_sae(sae, features, 
                                     num_epochs=100, 
                                     lr=0.0005,  # Reduced learning rate for stability
                                     batch_size=32,  # Increased batch size
                                     device=device,
                                     seed=GLOBAL_SEED)
# ... existing code ...
    # Save SAE model
    torch.save(sae.state_dict(), 'sae_model.pth')
    print("SAE model saved as 'sae_model.pth'")
    
    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('SAE Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sae_training_loss.png', dpi=300)
    plt.show()
    print("Training loss curve saved as 'sae_training_loss.png'")
    
    # ==========================================
    # Step 3: Visualize SAE Dictionary Atoms
    # ==========================================
    print("\n" + "="*60)
    print("Step 3: Visualizing SAE Dictionary Atoms...")
    print("="*60)
    visualize_dictionary(sae, features.shape[1:], class_names, num_atoms=20, device=device)
    
    # ==========================================
    # Step 4: Analyze Sparse Activation Patterns for Each Category
    # ==========================================
    print("\n" + "="*60)
    print("Step 4: Analyzing Class-Specific Activations...")
    print("="*60)
    class_activations = analyze_sparse_activations(sae, features, labels, class_names, device=device)
    
    # ==========================================
    # Step 5: Analyze Reconstruction Quality
    # ==========================================
    print("\n" + "="*60)
    print("Step 5: Analyzing Reconstruction Quality...")
    print("="*60)
    
    # Use fixed seed for reproducible sampling
    rng = np.random.RandomState(GLOBAL_SEED)
    sample_indices = rng.choice(len(features), size=5, replace=False)
    
    mse_scores = analyze_reconstruction_quality(sae, features, sample_indices, device=device)
    print(f"Average reconstruction MSE: {np.mean(mse_scores):.4f}")
    
    # ==========================================
    # Step 6: Feature-Category Correlation Heatmap
    # ==========================================
    print("\n" + "="*60)
    print("Step 6: Creating Feature-Class Correlation Heatmap...")
    print("="*60)
    
    plt.figure(figsize=(12, 10))
    # Only show top-50 most active atoms
    top_atoms = np.argsort(np.std(class_activations, axis=0))[-50:]
    heatmap_data = class_activations[:, top_atoms]
    
    sns.heatmap(heatmap_data, cmap='viridis', 
                xticklabels=[f'Atom {i}' for i in top_atoms],
                yticklabels=class_names,
                cbar_kws={'label': 'Activation Strength'})
    
    plt.title('SAE Atom Activation Patterns Across Classes', fontsize=14)
    plt.xlabel('SAE Dictionary Atoms', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('sae_feature_class_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Feature-class heatmap saved as 'sae_feature_class_heatmap.png'")
    
     # ==========================================
    # Step 7: Explore Specific Neuron Semantics
    # ==========================================
    print("\n" + "="*60)
    print("Step 7: Exploring Semantics of Neuron 6561...")
    print("="*60)
    
    # Create a loader for the full dataset to scan for max activations
    # Use val_transforms to match the feature extraction process
    scan_dataset = MyDataset(all_images, all_labels, transform=val_transforms)
    
    # Create generator with fixed seed for reproducibility
    scan_generator = torch.Generator()
    scan_generator.manual_seed(GLOBAL_SEED)
    scan_loader = DataLoader(scan_dataset, batch_size=2, shuffle=False, generator=scan_generator)
    
    # Explore neuron 6561
    # explore_neuron_semantics(sae, model, scan_loader, device, neuron_idx=19385, class_names=class_names)
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("SAE Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("1. sae_model.pth - Trained SAE model")
    print("2. sae_training_loss.png - Training loss curve")
    print("3. sae_dictionary_atoms.png - Dictionary atoms distribution")
    print("4. sae_class_activations.png - Class-specific activations")
    print("5. sae_reconstruction_quality.png - Reconstruction quality analysis")
    print("6. sae_feature_class_heatmap.png - Feature-class correlation heatmap")
    print("\nKey insights:")
    print("- SAE learns a sparse dictionary of semantic concepts")
    print("- Each atom represents a specific feature pattern")
    print("- Different classes activate different subsets of atoms")
    print("- This helps interpret what the CNN has learned")