#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import cv2

# 导入你原有的类和方法
from h6 import MyDataset, SimpleCNN, split_indices, val_transforms

# --------------------------
# 1. Grad-CAM 实现类
# --------------------------
class GradCAM:
    def __init__(self, model, target_layer_name='conv3'):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # 获取目标层
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")
        
        def save_gradients(module, grad_in, grad_out):
            # 使用 grad_out[0] 获取梯度
            self.gradients = grad_out[0]
        
        def save_activations(module, input, output):
            self.activations = output
            
        # 注册钩子
        self.target_layer.register_forward_hook(save_activations)
        # 使用 register_full_backward_hook 以兼容新版 PyTorch
        self.target_layer.register_full_backward_hook(save_gradients)

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

# --------------------------
# 2. 特征提取函数 (用于 t-SNE)
# --------------------------
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            # 提取 conv3 pool 之后的特征 (即进入全连接层之前的特征)
            x = model.pool(F.relu(model.conv1(images)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            x = x.view(x.size(0), -1) # 展平
            features.append(x.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.vstack(features), np.concatenate(labels)

# --------------------------
# 3. 主程序
# --------------------------
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cpu') # 默认 CPU，尝试 DirectML
    try:
        import torch_directml as dm
        device = dm.device()
        device = torch.device('cpu')
        print(f"Using DirectML device: {device}")
    except:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {device}")
        else:
            print("Using CPU")

    # 加载数据
    if not os.path.exists("all_images.npy") or not os.path.exists("all_labels.npy"):
        raise FileNotFoundError("Please run h6.py first to generate all_images.npy and all_labels.npy")
        
    all_images = np.load("all_images.npy")
    all_labels = np.load("all_labels.npy")
    full_dataset = MyDataset(images=all_images, labels=all_labels, transform=None)
    
    # 使用与训练时相同的随机种子和划分逻辑，确保测试集一致
    np.random.seed(42) 
    train_idx, val_idx, test_idx = split_indices(full_dataset)
    
    test_dataset = Subset(MyDataset(all_images, all_labels, val_transforms), test_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    class_names = sorted([d for d in os.listdir('Dataset') if os.path.isdir(os.path.join('Dataset', d))])

    # 加载模型
    model = SimpleCNN(num_classes=8).to(device)
    if not os.path.exists('best_custom_cnn.pth'):
        raise FileNotFoundError("Please run h6.py first to train the model and save best_custom_cnn.pth")
    
    # 加载权重时指定 map_location 以兼容不同设备
    model.load_state_dict(torch.load('best_custom_cnn.pth', map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # ==========================================
    # 任务一：t-SNE 特征可视化
    # ==========================================
    print("Extracting features for t-SNE...")
    features, true_labels = extract_features(model, test_loader, device)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings = tsne.fit_transform(features)
    
    df_tsne = pd.DataFrame(embeddings, columns=['x', 'y'])
    df_tsne['label'] = [class_names[int(l)] for l in true_labels]
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='label', palette='viridis', s=50, alpha=0.7)
    plt.title('t-SNE Visualization of Test Set Features')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tsne_analysis.png', dpi=300)
    plt.show()
    print("t-SNE plot saved as 'tsne_analysis.png'")

    # ==========================================
    # 任务二：Grad-CAM 热力图分析
    # ==========================================
    print("Generating Grad-CAM visualizations...")
    cam_extractor = GradCAM(model, target_layer_name='conv3')
    
    # 获取一个 batch 的图片和标签
    sample_images, sample_labels = next(iter(test_loader))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for i in range(8):
        img_tensor = sample_images[i].unsqueeze(0).to(device)
        true_label = sample_labels[i].item()
        
        # 获取预测结果
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred_label = torch.max(outputs, 1)
            pred_label = pred_label.item()
        
        # 生成 CAM
        cam = cam_extractor.generate_cam(img_tensor, pred_label)
        
        # 处理原始图片用于显示
        img_np = sample_images[i].permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # 叠加热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        overlay = 0.5 * img_np + 0.5 * (heatmap / 255.0)
        
        axes[i].imshow(overlay)
        status = "Correct" if pred_label == true_label else "Wrong"
        color = "green" if pred_label == true_label else "red"
        axes[i].set_title(f"T:{class_names[true_label]} P:{class_names[pred_label]}\n({status})", color=color)
        axes[i].axis('off')
        
    plt.suptitle('Grad-CAM Analysis on Test Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('gradcam_analysis.png', dpi=300)
    plt.show()
    print("Grad-CAM plot saved as 'gradcam_analysis.png'")