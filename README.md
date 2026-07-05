# Medical Image Classification Experiment - Complete Project Documentation

## 📋 Project Overview

This project is a PyTorch-based medical image classification system that uses a custom CNN model to classify 8 categories of medical images. The project includes a complete training pipeline, model evaluation, interpretability analysis (Grad-CAM, t-SNE), and Sparse Autoencoder (SAE) feature semantic analysis.

---

## 🗂️ Project Structure

```
h7/
├── cnn.py                                          # Main training script (also named as h6.py)
├── explanation1.py                                 # Grad-CAM and t-SNE visualization script
├── sae_analysis.py                                 # SAE feature semantic analysis script
├── Dataset/                                        # Raw dataset directory (8 classes)
├── all_images.npy                                  # Preprocessed image data
├── all_labels.npy                                  # Corresponding label data
├── best_custom_cnn.pth                             # Best trained model checkpoint
├── sae_model.pth                                   # Trained SAE model
├── *.png                                           # Generated visualization results
└── README.md                                       # This documentation file
```

**Note**: The file `cnn.py` is the main training script, also referred to as `h6.py` in the code comments. All two names refer to the same file.

---

## 🔧 Dependencies

### Python Packages

```bash
pip install torch torchvision
pip install numpy matplotlib pillow tqdm
pip install scikit-learn seaborn pandas opencv-python
pip install torch-directml  # Optional: Windows DirectML support
```

### Hardware Requirements

- **CPU**: Minimum configuration, but slower training speed
- **GPU**: NVIDIA GPU (recommended) with CUDA support
- **Windows DirectML**: Supports AMD/Intel GPU acceleration
- **RAM**: 8GB or more recommended (SAE training requires significant memory)

---

## 🚀 Quick Start

### Step 1: Prepare the Dataset

Organize your medical image dataset in the following structure:

```
Dataset/
├── class_0/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class_1/
│   ├── image1.jpg
│   └── ...
├── ...
└── class_7/
    ├── image1.jpg
    └── ...
```

**Note**: Ensure there are 8 class folders, each containing images (.jpg, .png, .jpeg formats) of the corresponding class.

### Step 2: Generate Preprocessed Data

Run the main training script to automatically generate `all_images.npy` and `all_labels.npy`:

```bash
python cnn.py
```

### Step 3: Train the Model

The main script will automatically perform the following operations:
1. Load and preprocess data
2. Split into training set (70%), validation set (15%), and test set (15%)
3. Train the SimpleCNN model (100 epochs with early stopping)
4. Save the best model to `best_custom_cnn.pth`
5. Plot training curves to `training_history.png`

### Step 4: Model Evaluation and Visualization

#### 4.1 Grad-CAM and t-SNE Analysis

```bash
python 解释1.py
```

Generated files:
- `tsne_analysis.png`: t-SNE feature dimensionality reduction visualization
- `gradcam_analysis.png`: Grad-CAM heatmap analysis

#### 4.2 SAE Feature Semantic Analysis

```bash
python sae_analysis.py
```

Generated files:
- `sae_model.pth`: Trained SAE model
- `sae_training_loss.png`: SAE training loss curve
- `sae_dictionary_atoms.png`: Dictionary atom distribution
- `sae_class_activations.png`: Class-specific activation patterns
- `sae_reconstruction_quality.png`: Reconstruction quality analysis
- `sae_feature_class_heatmap.png`: Feature-class correlation heatmap
- `sae_neuron_{idx}_activations.png`: Specific neuron activation visualization
- `sae_neuron_{idx}_attention.png`: Neuron attention map

---

## 📊 Model Architecture

### SimpleCNN Model

```
Input (3×224×224)
    ↓
Conv2d(3→16, 3×3) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d(16→32, 3×3) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool
    ↓
Flatten (64×28×28 = 50176)
    ↓
Linear(50176→256) + ReLU + Dropout(0.5)
    ↓
Linear(256→8)
    ↓
Output (8 classes)
```

### Key Features

- **Data Augmentation**: Random rotation (±50°), horizontal flip, color jitter, circular crop
- **Regularization**: BatchNorm, Dropout, Early Stopping
- **Optimizer**: AdamW (lr=0.001)
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss Function**: CrossEntropyLoss

---

## 🔬 Experimental Features

### 1. Ablation Study

In the main script (`cnn.py`), you can enable/disable the label shuffling experiment by commenting/uncommenting line 447:

```python
# Enable ablation study (shuffle training set labels)
ablation_train_dataset = Subset(MyDataset(all_images, shuffled_labels, train_transforms), train_idx)
train_dataset = ablation_train_dataset

# Disable ablation study (use normal labels)
train_dataset = Subset(MyDataset(all_images, all_labels, train_transforms), train_idx)
```

### 2. Circular Crop

Control the size of the circular region by adjusting the `radius_ratio` parameter in `CircularCrop`:

```python
# Use in transform
CircularCrop(radius_ratio=0.7)  # Radius is 70% of the short side
```

### 3. Stratified Sampling

Use the `split_indices()` function to ensure consistent class proportions across training/validation/test sets, avoiding class imbalance issues.

---

## 📈 Evaluation Metrics

### Classification Report

Includes for each class:
- **Precision**
- **Recall**
- **F1-Score**
- **Support** (number of samples)

### Confusion Matrix

Visualizes the correspondence between predicted and true labels.

### ROC Curve

Plots ROC curves for each class and calculates AUC values.

---

## 🎯 SAE (Sparse Autoencoder) Analysis

### Core Concept

SAE decomposes high-dimensional features extracted by CNN into sparse interpretable components, helping to understand what semantic concepts the model has learned.

### Top-K Hard Sparsity

The project adopts a Top-K hard sparsity strategy:
- Retains the K neurons with highest activation values (default K=64, accounting for 5% of total)
- Forces other neurons to zero
- Easier to control sparsity compared to traditional KL divergence soft sparsity

### Analysis Dimensions

1. **Dictionary Atom Visualization**: Displays basic feature patterns learned by SAE
2. **Class-Specific Activation**: Analyzes neuron combinations activated by different classes
3. **Reconstruction Quality**: Evaluates SAE's ability to preserve information
4. **Neuron Semantic Exploration**: Finds images that maximally activate specific neurons

---

## ⚙️ Hyperparameter Tuning Guide

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 77 | Batch size |
| num_epochs | 100 | Maximum training epochs |
| learning_rate | 0.001 | Initial learning rate |
| dropout | 0.5 | Dropout rate |
| early_stopping_patience | 10 | Early stopping patience |

### SAE Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 3136×8 | Hidden layer dimension (dictionary size) |
| k_value | 64 | Top-K sparsity |
| sparsity_weight | 0.01 | Sparsity weight (KL mode) |
| batch_size | 32 | SAE training batch size |
| lr | 0.0005 | SAE learning rate |

### Data Augmentation Parameters

```python
transforms.RandomRotation(50)  # Rotation angle range
transforms.ColorJitter(brightness=0.5, contrast=0.5, 
                       saturation=0.5, hue=0.5)  # Color jitter intensity
CircularCrop(radius_ratio=0.7)  # Circular crop ratio
```

---

## 🐛 Troubleshooting

### Q1: Out of Memory Error

**Solutions**:
- Reduce `batch_size` (e.g., from 77 to 32)
- Enable `use_extra_pooling=True` in SAE training to reduce feature dimensions
- Decrease SAE's `hidden_dim`

### Q2: Slow Training Speed

**Solutions**:
- Install CUDA version of PyTorch (if you have NVIDIA GPU)
- Or use `torch-directml` for AMD/Intel GPU support
- Set `num_workers > 0` to enable multi-process data loading

### Q3: NaN Loss Values

**Solutions**:
- Reduce learning rate (e.g., from 0.001 to 0.0005)
- Check if data normalization is correct
- Gradient clipping is already enabled

### Q4: Data Files Not Found

**Error**: `FileNotFoundError: Please run h6.py first...`

**Solution**: Run the main training script first to generate `.npy` files and the model.

---

## 📝 References

- PyTorch Official Documentation: https://pytorch.org/
- Grad-CAM Paper: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- t-SNE Visualization: [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- Sparse Autoencoder: [Online Learning of Invariant Feature Detectors](https://cs.stanford.edu/~quocle/LeZouNgLamCoatesNg12.pdf)

---

## 📄 License

This project is for academic research and educational purposes only.

---

## 📧 Contact

For questions or suggestions, please contact:
- Email: [jixiang2023@tmu.edu.cn]

---

**Last Updated**: 2026-07-05
