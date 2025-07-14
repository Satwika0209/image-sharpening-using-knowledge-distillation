# image-sharpening-using-knowledge-distillation
# 🧠 SwinIR Super-Resolution on REDS Dataset

This project applies the SwinIR-M model for ×2 Super-Resolution on the REDS video dataset using PyTorch in Google Colab. It optionally supports knowledge distillation to train a lightweight student CNN model.

---

## 🚀 Project Overview

- **Teacher Model:** SwinIR-M (x2 upscale) with pretrained weights from DF2K
- **Dataset:** [REDS Video Super-Resolution Dataset](https://www.kaggle.com/datasets/cookiemonsteryum/reds-video-superresolution-toy-dataset)
- **Framework:** PyTorch
- **Optional:** Knowledge Distillation setup for student network

---
🏁 Quick Start (in Colab)
Upload:

models/ folder (from SwinIR GitHub)

Pretrained weights: 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth

REDS dataset: reds_split_dataset/

Load model in inference.ipynb:

python
Copy
Edit
from models.network_swinir import SwinIR
import torch

model = SwinIR(
    upscale=2,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
)

checkpoint = torch.load('001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth')
model.load_state_dict(checkpoint['params'], strict=True)
model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
🧪 Dataset Description
The reds_split_dataset directory contains 3 subfolders:

train/LR and train/HR

val/LR and val/HR

test/LR and test/HR

Each contains image pairs for super-resolution.

📈 Evaluation Metrics
PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

