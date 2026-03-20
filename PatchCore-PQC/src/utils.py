# 在文件开头确保已导入所有必要的库
import csv
import logging
import os
import random
from turtle import right
from torch.nn.functional import threshold
import tqdm
from skimage import measure

import PIL
import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


import torch
import torch.nn as nn
import math

class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size: int, sigma: float):
        super(GaussianBlurLayer, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.weight = self._create_gaussian_kernel(kernel_size, sigma)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # 创建高斯核
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        y = torch.arange(kernel_size) - (kernel_size - 1) / 2
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        gaussian_2d = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
        return nn.Parameter(gaussian_2d, requires_grad=False)

    def forward(self, x):
        padding = self.kernel_size // 2
        return nn.functional.conv2d(x, self.weight, padding=padding, groups=x.shape[1])

def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=2,
):
    """Generate anomaly segmentation images with predicted masks."""
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    # 创建保存文件夹
    heatmap_folder = os.path.join(savefolder, "heatmaps")
    predicted_mask_folder = os.path.join(savefolder, "predicted_masks")
    os.makedirs(heatmap_folder, exist_ok=True)
    os.makedirs(predicted_mask_folder, exist_ok=True)

    for image_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        try:
            # 读取原始图像并获取其分辨率
            original_image = PIL.Image.open(image_path).convert("RGB")
            original_width, original_height = original_image.size

            image = image_transform(original_image)
            if not isinstance(image, np.ndarray):
                image = image.numpy()

            # 规范化分割结果
            LOGGER.info(f"Segmentation stats for {image_path}: min={segmentation.min()}, max={segmentation.max()}, mean={segmentation.mean()}")
            segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min() + 1e-8)

            # 处理分割图，生成二值化预测mask
            # 优化：提高阈值以减少误判的区域
            threshold_value = 0.3  # 从0.1提高到0.3，可以根据实际效果进一步调整
            binary_mask = segmentation > threshold_value

            # 优化：增加小区域移除的最小面积
            from skimage.morphology import remove_small_objects, binary_opening, binary_closing
            binary_mask = remove_small_objects(binary_mask, min_size=100)  # 从50增加到100

            # 优化：添加形态学操作进一步细化掩码
            # 先腐蚀再膨胀，有助于去除小噪声并连接断开的区域
            binary_mask = binary_opening(binary_mask, footprint=np.ones((3, 3)))
            binary_mask = binary_closing(binary_mask, footprint=np.ones((5, 5)))

            # 使用图像文件名生成唯一的保存路径
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # 将热力图和预测mask调整到原始图像的分辨率
            segmentation_pil = PIL.Image.fromarray((segmentation * 255).astype(np.uint8))
            resized_segmentation = segmentation_pil.resize((original_width, original_height), PIL.Image.BICUBIC)
            resized_segmentation = np.array(resized_segmentation) / 255.0

            # 调整二值化mask的大小
            binary_mask_pil = PIL.Image.fromarray(binary_mask.astype(np.uint8) * 255)
            resized_binary_mask = binary_mask_pil.resize((original_width, original_height), PIL.Image.NEAREST)

            # 保存单独的热力图
            save_path = os.path.join(heatmap_folder, f"{base_name}_heatmap.png")
            fig = plt.figure(figsize=(original_width / 100, original_height / 100), dpi=100)
            plt.imshow(original_image)
            plt.imshow(resized_segmentation, alpha=0.5, cmap='jet')
            plt.axis("off")
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # 保存预测的二值化mask
            mask_save_path = os.path.join(predicted_mask_folder, f"{base_name}_predicted_mask.png")
            resized_binary_mask.save(mask_save_path)

        except Exception as e:
            LOGGER.error(f"Error processing {image_path}: {str(e)}")
            continue


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    
    if mode not in ["iterate", "overwrite"]:
        raise ValueError(f"Invalid mode: {mode}. Supported modes are 'iterate' and 'overwrite'.")
    
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    LOGGER.info(f"Results will be saved to: {save_path}")
    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids) > 0:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 明确禁用 benchmark

def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
        column_names=None,
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if column_names is None:
        column_names = [
            "Instance AUROC",
            "Full Pixel AUROC",
            "Full PRO",
            "Anomaly Pixel AUROC",
            "Anomaly PRO",
        ]
    if row_names is not None and len(row_names) != len(results):
        raise ValueError(f"Number of row names ({len(row_names)}) does not match number of result rows ({len(results)}).")

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics