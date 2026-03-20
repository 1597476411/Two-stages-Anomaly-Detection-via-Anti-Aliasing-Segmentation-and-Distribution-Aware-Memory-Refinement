import os
import glob
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP ,U2NETP_LLPF

class SelectiveHighlightSuppressor(nn.Module):
    """
    Improved Version: Pure PyTorch + Adaptive Highlight Threshold
    - Automatically detect the brightest regions in each image as highlights
    - Only suppress strong highlights while preserving small dirt particles completely
    """

    def __init__(self, strength=1, window_size=61, kernel_size=15):
        super().__init__()
        self.strength = strength
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.register_buffer('blur_kernel', self._get_gaussian_kernel(kernel_size, sigma=3.0))

    def _get_gaussian_kernel(self, k, sigma):
        x = torch.linspace(-(k - 1) / 2, (k - 1) / 2, k)
        kern1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kern2d = torch.outer(kern1d, kern1d)
        return (kern2d / kern2d.sum()).view(1, 1, k, k)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        local_mean = F.avg_pool2d(gray, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)

        diff = gray - local_mean
        bright_mask = torch.sigmoid((diff - 0.1) * 20) * torch.sigmoid((gray - 0.6) * 10)

        if mask is not None:
            bright_mask = bright_mask * mask

        smoothed = F.conv2d(x, self.blur_kernel.expand(C, -1, -1, -1), padding=self.kernel_size // 2, groups=C)
        return x * (1 - bright_mask) + (smoothed * self.strength) * bright_mask

# ============================== Constant Configuration ==============================
MODEL_NAME = 'u2netp_llpf'
MODEL_WEIGHT_PATH = r"u2netp_llpf_best_model.pth"
INPUT_SIZE = 320
OUTPUT_SIZE = 1600

TEST_IMAGE_DIR = "test_image"
TEST_GT_DIR = "test_mask"
PREDICTION_DIR = "test_results"

BETA2_FOR_F2 = 4.0

# ============================== Boundary & Affine Functions (Unchanged) ==============================
def extract_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    diag_len = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))
    eroded = binary_erosion(mask, iterations=dilation)
    return (mask ^ eroded).astype(np.uint8)

def calculate_boundary_fscore(pred_mask, gt_mask):
    pred_b = extract_boundary(pred_mask)
    gt_b = extract_boundary(gt_mask)
    tp = np.sum((pred_b == 1) & (gt_b == 1))
    fp = np.sum((pred_b == 1) & (gt_b == 0))
    fn = np.sum((pred_b == 0) & (gt_b == 1))
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)

def compute_mask_centroid(binary_mask):
    m = cv2.moments(binary_mask)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy

def build_translation_affine(cx, cy):
    target_c = OUTPUT_SIZE // 2
    tx = target_c - cx
    ty = target_c - cy
    return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

def apply_affine(image, mask, M):
    img_a = cv2.warpAffine(image, M, (OUTPUT_SIZE, OUTPUT_SIZE),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_a = cv2.warpAffine(mask, M, (OUTPUT_SIZE, OUTPUT_SIZE),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_a, mask_a

# ============================== Utility Functions ==============================
def normalize_prediction(pred):
    return (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

def resize_mask(mask, size):
    return np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (size[1], size[0]), Image.BILINEAR
        )
    ) / 255.0

def apply_mask(image, mask):
    return (image * mask[..., None]).astype(np.uint8)

def calculate_metrics(pred, gt):
    pred_b = (pred >= 0.5).astype(np.uint8)
    gt_b = (gt >= 0.5).astype(np.uint8)
    tp = np.sum((pred_b == 1) & (gt_b == 1))
    fp = np.sum((pred_b == 1) & (gt_b == 0))
    fn = np.sum((pred_b == 0) & (gt_b == 1))
    iou = tp / (tp + fp + fn + 1e-8)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f2 = (1 + BETA2_FOR_F2) * p * r / (BETA2_FOR_F2 * p + r + 1e-8)
    bf = calculate_boundary_fscore(pred_b, gt_b)
    return iou, f2, bf

# ============================== Main Process ==============================
def main():
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    img_list = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*")))
    gt_list = sorted(glob.glob(os.path.join(TEST_GT_DIR, "*"))) if TEST_GT_DIR else None

    dataset = SalObjDataset(
        img_list,
        gt_list if gt_list else [],
        transform=transforms.Compose([RescaleT(INPUT_SIZE), ToTensorLab(flag=0)])
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load U2NETP_LLPF model (with built-in highlight suppression)
    net = U2NETP_LLPF(3, 1) if MODEL_NAME == 'u2netp' else U2NET(3, 1)
    net.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device), strict=False)
    net.to(device).eval()
    ious, f2s, bfs, times = [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            img = data['image'].to(device).float()
            orig_np = np.array(Image.open(img_list[i]).convert("RGB"))

            t0 = time.time()
            d0, *_ = net(img)
            times.append((time.time() - t0) * 1000)

            pred_320 = d0.squeeze().cpu().numpy()
            pred_320 = np.clip(pred_320, 0, 1)

            mask_orig = resize_mask(pred_320, orig_np.shape[:2])
            binary = (mask_orig >= 0.5).astype(np.uint8)
            centroid = compute_mask_centroid(binary)

            if centroid is not None:
                M = build_translation_affine(*centroid)
                orig_aligned, binary_aligned = apply_affine(orig_np, binary * 255, M)
                mask_aligned = binary_aligned / 255.0
            else:
                orig_aligned = cv2.resize(orig_np, (OUTPUT_SIZE, OUTPUT_SIZE))
                mask_aligned = cv2.resize(mask_orig, (OUTPUT_SIZE, OUTPUT_SIZE))

            orig_aligned_roi = apply_mask(orig_aligned, mask_aligned)

            # ====================== Core Modification: Manual Suppression Comparison ======================
            # Convert original ROI to Tensor for suppressor processing
            roi_tensor = torch.from_numpy(orig_aligned_roi).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            feedback_tensor = torch.from_numpy(mask_aligned).float().unsqueeze(0).unsqueeze(0).to(device)

            clean_tensor = net.suppressor(roi_tensor, feedback_mask=feedback_tensor)

            clean_roi = (clean_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            combined = np.hstack((orig_aligned_roi, clean_roi))
            Image.fromarray(combined).save(os.path.join(PREDICTION_DIR, f"{i:04d}_COMPARE.png"))

            if TEST_GT_DIR:
                gt = data['label'].squeeze().numpy()
                iou, f2, bf = calculate_metrics(pred_320, gt)
                ious.append(iou)
                f2s.append(f2)
                bfs.append(bf)

    print("-" * 30)
    print(f"IoU: {np.mean(ious):.4f}")
    print(f"F2: {np.mean(f2s):.4f}")
    print(f"Boundary F: {np.mean(bfs):.4f}")
    print(f"Avg Time: {np.mean(times):.2f} ms")

if __name__ == "__main__":
    main()