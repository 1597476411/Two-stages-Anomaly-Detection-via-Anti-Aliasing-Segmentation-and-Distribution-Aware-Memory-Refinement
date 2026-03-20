import contextlib
import logging
import os
import sys
import time
import random
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms

# sys.path.append('src')
import src.backbones as backbones
import src.common as common
import src.metrics as metrics
import src.sampler as sampler
import src.utils as utils
import src.softpatch as softpatch
import src.datasets as datasets

# Disable duplicate KMP library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logger
LOGGER = logging.getLogger(__name__)

# Dataset configuration
_DATASETS = {
    "mvtec": ["src.datasets.mvtec", "MVTecDataset"],
    "btad": ["src.datasets.btad", "BTADDataset"],
    "bowel": ["src.datasets.bowel", "CustomDataset"],
    "mvtec2": ["src.datasets.mvtec2", "MVTecDataset"],
    "dagm2007": ["src.datasets.dagm2007", "DAGM2007Dataset"],
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PQC/RGBPatchcore')
    
    # Project configuration
    parser.add_argument('--gpu', type=int, default=[], action='append')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_path", type=str, default="result")
    parser.add_argument("--log_project", type=str, default="project")
    parser.add_argument("--log_group", type=str, default="group")
    parser.add_argument("--save_segmentation_images", action='store_true')
    
    # Backbone configuration
    parser.add_argument("--backbone_names", "-b", type=str, nargs='*', default=['DINO_VIT-B/16'])
    parser.add_argument("--layers_to_extract_from", "-le", type=str, nargs='*', action='append', default=None)
    
    # Coreset sampler configuration
    parser.add_argument("--sampler_name", type=str, default="approx_greedy_coreset")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--faiss_on_gpu", action='store_true')
    parser.add_argument("--faiss_num_workers", type=int, default=4)
    parser.add_argument("--disable_patch_removal", action="store_true", help="Disable patch removal logic")
    parser.add_argument("--weight_method", type=str, default="lof")
    parser.add_argument("--lof_k", type=int, default=6)
    parser.add_argument("--without_soft_weight", action='store_true')
    parser.add_argument("--enable_normal_patch_removal", action="store_true", help="Enable normal patch removal strategy")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True, choices=["mvtec", "btad", "bowel", 'mvtec2', 'dagm2007'])
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--subdatasets", "-d", action='append', type=str, default=None, help="Subdataset name")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--overlap", action='store_true')
    parser.add_argument("--noise_augmentation", action='store_true')
    parser.add_argument("--fold", type=int, default=0)
    
    args = parser.parse_args()
    
    # Process layers_to_extract_from
    if args.layers_to_extract_from is None or len(args.layers_to_extract_from) == 0:
        args.layers_to_extract_from = ['layer2', 'layer3']
    else:
        cleaned_layers = []
        for layer_input in args.layers_to_extract_from:
            # Convert input to string and remove extra characters
            layer_str = str(layer_input).strip("[]'\"")
            if ',' in layer_str:
                # Split comma-separated layer names
                split_layers = [l.strip() for l in layer_str.split(',') if l.strip()]
                cleaned_layers.extend(split_layers)
            else:
                if layer_str:
                    cleaned_layers.append(layer_str)
        args.layers_to_extract_from = cleaned_layers

    LOGGER.info(f"Parsed layers_to_extract_from: {args.layers_to_extract_from}")
    return args


def get_dataloaders(args):
    """Create training and testing dataloaders"""
    data_path = args.data_path
    batch_size = args.batch_size
    resize = args.resize
    imagesize = args.imagesize
    noise = args.noise
    overlap = args.overlap
    noise_augmentation = args.noise_augmentation
    fold = args.fold

    dataset_info = _DATASETS[args.dataset]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataloaders = []
    
    subdatasets = args.subdatasets if args.subdatasets else [None]
 
    train_transform = transforms.Compose([
        transforms.Resize((imagesize, imagesize), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
        
    for subdataset in subdatasets:
        subdataset_name = subdataset if subdataset is not None else "all"

        # Create training dataset
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            source=data_path,
            classname=subdataset,
            transform=train_transform,
            split=dataset_library.DatasetSplit.TRAIN,
        )

        # Create testing dataset
        test_dataset = dataset_library.__dict__[dataset_info[1]](
            source=data_path,
            classname=subdataset,
            transform=train_transform,
            split=dataset_library.DatasetSplit.TEST,
        )

        # Add noise to training dataset if specified
        if noise > 0:
            anomaly_index = [idx for idx in range(len(test_dataset)) if test_dataset[idx]["is_anomaly"]]
            train_length = len(train_dataset)
            noise_number = int(noise * train_length)

            if noise_number > len(anomaly_index):
                LOGGER.warning(f"Requested {noise_number} noisy samples, only {len(anomaly_index)} available.")
                noise_index = random.choices(anomaly_index, k=noise_number)
            else:
                noise_index_path = Path(f"noise_index/{args.dataset}_noise{noise}_fold{fold}")
                noise_index_path.mkdir(parents=True, exist_ok=True)
                path = noise_index_path / f"{subdataset_name}-noise{noise}.pth"

                if path.exists():
                    noise_index = torch.load(path)
                else:
                    noise_index = random.sample(anomaly_index, noise_number)
                    torch.save(noise_index, path)

            # Create noise dataset and add to training data
            noise_dataset = Subset(test_dataset, noise_index)
            if noise_augmentation:
                noise_dataset = datasets.NoiseDataset(noise_dataset)

            train_dataset = ConcatDataset([train_dataset, noise_dataset])

            # Remove noisy samples from test set if overlap is disabled
            if not overlap:
                new_test_idx = list(set(range(len(test_dataset))) - set(noise_index))
                test_dataset = Subset(test_dataset, new_test_idx)

        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        )

        # Set dataset name
        train_dataloader.name = args.dataset
        if subdataset is not None:
            train_dataloader.name += "_" + subdataset

        dataloaders.append({
            "training": train_dataloader,
            "testing": test_dataloader,
        })

    return dataloaders


def get_sampler(sampler_name, sampling_ratio, device):
    """Get coreset sampler based on sampler name"""
    if sampler_name == "identity":
        return sampler.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        return sampler.GreedyCoresetSampler(sampling_ratio, device)
    elif sampler_name == "approx_greedy_coreset":
        return sampler.ApproximateGreedyCoresetSampler(sampling_ratio, device)


def get_coreset(args, imagesize, sampler, device):
    """Initialize coreset instances with specified backbones"""
    input_shape = (3, imagesize, imagesize)
    backbone_names = list(args.backbone_names)
    
    if len(backbone_names) > 1:
        LOGGER.warning(f"Multiple backbones specified: {backbone_names}. Only the first backbone will be used.")
        backbone_names = backbone_names[:1]    
        
    layers_to_extract_from = args.layers_to_extract_from
    
    if any('vit' in name.lower() or 'deit' in name.lower() for name in backbone_names):
        layers_to_extract_from = args.layers_to_extract_from if args.layers_to_extract_from else ['4']
        layers_to_extract_from = [
            str(int(layer.split('.')[1])) if '.' in layer and layer.startswith('blocks.') else layer
            for layer in layers_to_extract_from
        ]
        try:
            indices = [int(layer) for layer in layers_to_extract_from]
            max_blocks = 12
            for idx in indices:
                if idx >= max_blocks:
                    raise ValueError(f"Layer index {idx} exceeds {max_blocks} blocks in ViT")
        except ValueError:
            raise ValueError(f"ViT layers must be integer indices, got {layers_to_extract_from}")
    else:
        layers_to_extract_from = args.layers_to_extract_from if args.layers_to_extract_from else ['layer2', 'layer3']
    
    layers_to_extract_from_coll = [layers_to_extract_from for _ in backbone_names]

    nn_method = common.FaissNN(
        args.faiss_on_gpu, 
        args.faiss_num_workers, 
        device=device.index if device.type == "cuda" else None
    )
    
    loaded_coresets = []
    for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
        LOGGER.info(f"Loading backbone: {backbone_name}, Layers: {layers_to_extract_from}")

        # Load backbone model
        backbone = backbones.load(
            backbone_name,
            seed=args.seed,    
        )

        backbone.name = backbone_name
        
        coreset_instance = softpatch.SoftPatch(device)
        coreset_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            featuresampler=sampler,
            nn_method=nn_method,
            disable_patch_removal=args.disable_patch_removal,
            LOF_k=args.lof_k,
            weight_method=args.weight_method,
            soft_weight_flag=not args.without_soft_weight,
            enable_normal_patch_removal=args.enable_normal_patch_removal
        )
        loaded_coresets.append(coreset_instance)
    
    return loaded_coresets


def run(args):
    seed = args.seed
    
    run_save_path = utils.create_storage_folder(
        args.results_path, args.log_project, args.log_group, mode="iterate"
    )

    list_of_dataloaders = get_dataloaders(args)

    device = utils.set_torch_device(args.gpu)
    LOGGER.info(f"Using device: {device}")
    
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    # Process each dataset
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        dataset_name = dataloaders["training"].name
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        
        start_time = time.time()
        utils.fix_seeds(seed, device)

        with device_context:
            torch.cuda.empty_cache()
            
            sampler_instance = get_sampler(args.sampler_name, args.sampling_ratio, device)
            coreset_list = get_coreset(args, args.imagesize, sampler_instance, device)
                  
            train_time = 0
            for i, coreset in enumerate(coreset_list):
                torch.cuda.empty_cache()
                
                if coreset.backbone.seed is not None:
                    utils.fix_seeds(coreset.backbone.seed, device)
                
                LOGGER.info(f"Training models ({i + 1}/{len(coreset_list)})")
                train_start_time = time.time()
                coreset.fit(dataloaders["training"])
                train_end_time = time.time()
                train_time = train_end_time - train_start_time
                LOGGER.info(f"Training time: {train_time:.2f} seconds")

                coreset_patch_count = len(coreset.memory_bank) if hasattr(coreset, 'memory_bank') else 0
                feature_dim = coreset.memory_bank.shape[1] if hasattr(coreset, 'memory_bank') and len(coreset.memory_bank.shape) > 1 else 0
                LOGGER.info(f"Coreset patch count: {coreset_patch_count}")
                LOGGER.info(f"Feature dimension: {feature_dim}")
            
            aggregator = {"scores": [], "segmentations": [], "inference_times": []}
            
            for i, coreset in enumerate(coreset_list):
                torch.cuda.empty_cache()
                LOGGER.info(f"Embedding test data with models ({i + 1}/{len(coreset_list)})")
                
                scores, segmentations, labels_gt, masks_gt = [], [], [], []
                inference_times = []
                test_loader = dataloaders["testing"]
            
                if len(test_loader) == 0:
                    LOGGER.error("Test loader is empty. Please check the dataset or data loading process.")
                    return
                
                for batch in test_loader:
                    start_time_batch = time.time()
                    batch_scores, batch_segmentations = coreset.predict(batch)
                    end_time_batch = time.time()
                    
                    batch_time = (end_time_batch - start_time_batch) * 1000 / len(batch["image"])
                    inference_times.extend([batch_time] * len(batch["image"]))
                    
                    scores.extend(batch_scores)
                    
                    if len(batch_segmentations) == 0:
                        LOGGER.error("Coreset prediction returned empty segmentations for the current batch.")
                        continue
                    
                    segmentations.extend(batch_segmentations)
                    labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                    masks_gt.extend(batch["mask"].numpy().tolist())

                if len(segmentations) == 0:
                    LOGGER.error("Segmentations array is empty. Skipping normalization.")
                    return
                
                for batch in test_loader:
                    LOGGER.info(f"Batch is_anomaly: {batch['is_anomaly']}")
                    break

                segmentations = np.array(segmentations)
                segmentations = (segmentations - np.min(segmentations)) / (np.max(segmentations) - np.min(segmentations) + 1e-5)

                masks_gt = np.array(masks_gt)
                masks_gt = (masks_gt > 0).astype(np.uint8)
                
                num_positive_masks = sum(np.sum(mask) > 0 for mask in masks_gt)
                
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
                aggregator["inference_times"].append(inference_times)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores + 1e-5)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            inference_times = np.array(aggregator["inference_times"])
            avg_inference_time = np.mean(inference_times)

            test_end = time.time()
            LOGGER.info(f"Training time:{train_time:.2f} seconds, Testing time:{test_end - train_start_time:.2f} seconds")

            if args.save_segmentation_images:
                LOGGER.info("Saving segmentation images...")
                indices = dataloaders["testing"].dataset.indices
                data_to_iterate = dataloaders["testing"].dataset.dataset.data_to_iterate

                image_paths = []
                mask_paths = []
                
                for i in indices:
                    try:
                        item = data_to_iterate[i]
                        
                        if isinstance(item, str):
                            img_path = item
                        elif hasattr(item, 'tolist'):
                            item_list = item.tolist()
                            img_path = None
                            for element in item_list:
                                if isinstance(element, str) and element.startswith('datasets/'):
                                    img_path = element
                                    break
                            
                            if img_path is None:
                                img_path = f"unknown_image_{i}.png"
                                LOGGER.warning(f"Could not extract valid path from data structure for index {i}")
                        else:
                            img_path = str(item)
                            LOGGER.warning(f"Unexpected data type {type(item)} for index {i}")
                        
                        image_paths.append(img_path)
                        
                        try:
                            if 'test' in img_path:
                                mask_path = img_path.replace('test', 'ground_truth')
                                name_parts = mask_path.split('.')
                                if len(name_parts) > 1:
                                    mask_path = '.'.join(name_parts[:-1]) + '_mask.' + name_parts[-1]
                                else:
                                    mask_path += '_mask'
                            else:
                                mask_basename = os.path.basename(img_path)
                                mask_path = os.path.join("datasets/bowel/ground_truth", mask_basename)
                        except:
                            mask_path = os.path.join("datasets/bowel/ground_truth", f"unknown_mask_{i}.png")
                        
                        mask_paths.append(mask_path)
                        
                    except Exception as e:
                        LOGGER.error(f"Error processing index {i}: {str(e)}")
                        image_paths.append(f"unknown_image_{i}.png")
                        mask_paths.append(f"unknown_mask_{i}.png")

                for image_path in image_paths:
                    if isinstance(image_path, str) and not os.path.exists(image_path):
                        LOGGER.warning(f"Image file does not exist: {image_path}")

                for mask_path in mask_paths:
                    if isinstance(mask_path, str) and not os.path.exists(mask_path):
                        LOGGER.warning(f"Mask file does not exist: {mask_path}")

                if len(image_paths) == 0 or len(mask_paths) == 0:
                    LOGGER.error("Image paths or mask paths are empty. Please check the dataset.")
                    return

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                LOGGER.info(f"Saving segmentation images to {image_save_path}")
                
                utils.plot_segmentation_images(
                    savefolder=image_save_path,
                    image_paths=image_paths,
                    segmentations=segmentations,
                    anomaly_scores=scores,
                    mask_paths=mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )
                LOGGER.info(f"Saved segmentation images to: {image_save_path}")
                LOGGER.info(f"Number of images processed: {len(image_paths)}")
            
            LOGGER.info("Computing evaluation metrics.")
            
            anomaly_ground_truth_labels = np.array(labels_gt)
            anomaly_prediction_weights = np.ones_like(anomaly_ground_truth_labels, dtype=float)
            
            unique_labels = np.unique(anomaly_ground_truth_labels)
            if len(unique_labels) < 2:
                LOGGER.warning("anomaly_ground_truth_labels contains only one class. Adding dummy samples.")
                anomaly_ground_truth_labels = np.append(anomaly_ground_truth_labels, [0, 1])
                anomaly_prediction_weights = np.append(anomaly_prediction_weights, [0.0, 1.0])
            
            auroc = metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt
            )["auroc"]
            
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]
            full_pixel_pro = pixel_scores["pro"]

            # Compute pixel-wise metrics only for anomaly images
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            
            if sel_idxs:
                pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores["auroc"]
                anomaly_pixel_pro = pixel_scores["pro"]
            else:
                LOGGER.warning("No positive samples in masks_gt for anomaly_pixel_auroc.")
                anomaly_pixel_auroc = float('nan')
                anomaly_pixel_pro = float('nan')

            # Collect results
            result_collect.append({
                "dataset_name": dataset_name,
                "image_auroc": auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "full_pixel_pro": full_pixel_pro,
                "anomaly_pixel_pro": anomaly_pixel_pro,
                "anomaly_pixel_auroc": anomaly_pixel_auroc,
                "avg_inference_time_ms": avg_inference_time,
                "coreset_patch_count": coreset.patch_count if hasattr(coreset, 'patch_count') else 0,
                "feature_dim": coreset.feature_dim if hasattr(coreset, 'feature_dim') else 0,
            })

            # Log results
            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:4.4f}".format(key, item))

        LOGGER.info("\n\n-----\n")

    # Save final results
    if result_collect:
        result_metric_names = list(result_collect[-1].keys())[1:]
        result_dataset_names = [results["dataset_name"] for results in result_collect]
        result_scores = [list(results.values())[1:] for results in result_collect]
        
        utils.compute_and_store_final_results(
            run_save_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_dataset_names,
        )


if __name__ == "__main__":
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    
    args = parse_args()
    run(args)