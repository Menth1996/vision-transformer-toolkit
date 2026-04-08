# Vision Transformer Toolkit

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive toolkit for training, evaluating, and deploying Vision Transformers (ViT) for various computer vision tasks.

## Features
- Pre-trained ViT models (Base, Large, Huge)
- Custom data loading pipelines for ImageNet and custom datasets
- Distributed Data Parallel (DDP) training support
- Export to ONNX and TensorRT for optimized inference

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from vit_toolkit.models import ViT
from vit_toolkit.trainer import Trainer

model = ViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072)
trainer = Trainer(model, train_dataloader, val_dataloader, epochs=100, lr=3e-4)
trainer.train()
```
