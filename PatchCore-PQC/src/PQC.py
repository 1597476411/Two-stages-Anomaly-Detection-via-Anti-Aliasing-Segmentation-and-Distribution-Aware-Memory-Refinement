import logging
import os
import pickle
import tqdm
import torch
import numpy as np
import sys

from sklearn.neighbors import LocalOutlierFactor
import torch.nn.functional as F
from scipy.stats import skew, kurtosis

import faiss
# Assuming these are part of your package structure
from . import common, sampler, multi_variate_gaussian, backbones

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    """
    Cleaned and optimized PatchCore implementation.
    - All Chinese text converted to English
    - Soft-weighting completely removed
    - t-SNE visualization and all related code removed
    - Code structure simplified for public release
    """

    def __init__(self, device):
        super(PatchCore, self).__init__()
        self.device = device
        self.backbone = None
        self.seed = None

    def load(
        self,
        backbone,
        device,
        input_shape,
        layers_to_extract_from=("layer2", "layer3"),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler.ApproximateGreedyCoresetSampler(percentage=0.1, device=torch.device("cuda")),
        nn_method=common.FaissNN(False, 4),
        lof_k=5,
        weight_method="lof",
        disable_patch_removal=False,
        enable_normal_patch_removal=False,
        redundant_threshold=0.95,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.backbone.seed = self.seed
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator.to(self.device)

        self.anomaly_scorer = common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        # Weighted coreset sampler
        self.featuresampler = sampler.WeightedGreedyCoresetSampler(
            featuresampler.percentage, featuresampler.device
        )

        self.patch_weight = None
        self.feature_shape = []
        self.lof_k = lof_k
        self.coreset_weight = None
        self.weight_method = weight_method
        self.disable_patch_removal = disable_patch_removal
        self.enable_normal_patch_removal = enable_normal_patch_removal
        self.redundant_threshold = redundant_threshold

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            raw_features = self.forward_modules["feature_aggregator"](images)

        # Normalize different return types to list of tensors
        if isinstance(raw_features, dict):
            feature_list = list(raw_features.values())
        elif isinstance(raw_features, (list, tuple)):
            feature_list = raw_features
        else:
            feature_list = [raw_features]

        if not feature_list:
            raise RuntimeError("No features extracted from backbone")

        # Optional single feature map visualization (first layer, first image) — not t-SNE
        if feature_list:
            try:
                feat = feature_list[0][0].unsqueeze(0)
                resized_feature = F.interpolate(
                    feat, size=(14, 14), mode='bilinear', align_corners=False
                )
                resized_feature = resized_feature.squeeze(0).mean(dim=0).detach().cpu().numpy()
                normalized_feature = (
                    resized_feature - np.min(resized_feature)
                ) / (np.max(resized_feature) - np.min(resized_feature) + 1e-8)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                img = plt.imshow(normalized_feature, cmap='viridis', aspect='auto')
                plt.colorbar(img, shrink=0.8, aspect=20, pad=0.05, format='%.2f')
                plt.close()
            except Exception as e:
                LOGGER.warning(f"Feature visualization skipped: {e}")

        # Patchify all layers
        patched = [
            self.patch_maker.patchify(feat_map, return_spatial_info=True)
            for feat_map in feature_list
        ]
        patch_shapes = [p[1] for p in patched]
        features = [p[0] for p in patched]

        ref_num_patches = patch_shapes[0]

        # Align multi-layer features to the same spatial resolution
        if len(features) > 1:
            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])

                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # Preprocessing + aggregation
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """Compute embeddings and fill memory bank."""
        self._fill_memory_bank(training_data)

    def save_features_to_npy(self, features, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)
        LOGGER.info(f"Features saved to {save_path}")

    def remove_redundant_patches_cosine_gpu(self, patch_features: torch.Tensor, similarity_threshold: float = 0.95):
        """
        Remove redundant patches using cosine similarity (GPU accelerated).
        """
        if patch_features.dim() == 3:
            patch_features = patch_features.flatten(0, 1)  # [N*B, D]

        device = patch_features.device
        N = patch_features.shape[0]

        features = torch.nn.functional.normalize(patch_features, dim=1)
        batch_size = 1000
        keep_mask = torch.ones(N, dtype=torch.bool, device=device)

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_features = features[i:end_i]
            sim_matrix = batch_features @ features.T

            # Avoid self-similarity
            if i == end_i - batch_size:
                sim_matrix[:, i:end_i].fill_diagonal_(0.0)

            for j in range(i, end_i):
                if not keep_mask[j]:
                    continue
                redundant_idx = (sim_matrix[j - i] > similarity_threshold) & keep_mask
                redundant_idx[j] = False
                keep_mask[redundant_idx] = False

        kept_features = patch_features[keep_mask]
        LOGGER.info(f"Removed redundant patches: {N - keep_mask.sum().item()} / {N} | threshold={similarity_threshold}")
        return kept_features, keep_mask

    def _fill_memory_bank(self, input_data):
        """Compute support features and build memory bank with optional filtering stages."""
        LOGGER.info(f"Anomaly patch removal enabled: {not self.disable_patch_removal}")
        LOGGER.info(f"Normal (redundant) patch removal enabled: {self.enable_normal_patch_removal}")

        self.forward_modules.eval()

        def _image_to_features(input_image):
            if not isinstance(input_image, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(input_image)}")
            input_image = input_image.to(torch.float).to(self.device)
            with torch.no_grad():
                return self._embed(input_image)

        # Collect all features
        features_list = []
        total = len(input_data) if hasattr(input_data, "__len__") else None
        if total is None:
            input_data = list(input_data)
            total = len(input_data)

        with tqdm.tqdm(input_data, desc="Computing support features...", total=total) as iterator:
            for image in iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features_list.append(_image_to_features(image))

        features = np.concatenate(features_list, axis=0)

        with torch.no_grad():
            self.feature_shape = self._embed(
                image.to(torch.float).to(self.device), provide_patch_shapes=True
            )[1][0]
            LOGGER.info(f"Training patch shape: {self.feature_shape}")

            # Stage 1: Full features
            full_patch_weight = self._compute_patch_weight(features)
            full_patch_weight = full_patch_weight.reshape(-1)
            full_scores_np = full_patch_weight.cpu().numpy()

            stage1_weight_np = full_patch_weight.cpu().numpy().flatten()
            stage1_skew = float(skew(stage1_weight_np))
            stage1_kurt = float(kurtosis(stage1_weight_np))
            LOGGER.info(f"[Stage 1: Full features] skewness={stage1_skew:.4f}, kurtosis={stage1_kurt:.4f}")

            filtered_features = features
            current_indices = torch.arange(len(features), device=self.device)
            self.patch_weight = full_patch_weight.clone()

            # Stage 2: Anomaly (outlier) patch removal
            if not self.disable_patch_removal:
                weight_mean = torch.mean(self.patch_weight)
                weight_std = torch.std(self.patch_weight)
                upper_threshold = weight_mean + 1.5 * weight_std
                lower_threshold = weight_mean - 1.5 * weight_std

                sampling_weight = torch.where(
                    (self.patch_weight > upper_threshold) | (self.patch_weight < lower_threshold),
                    torch.zeros_like(self.patch_weight),
                    torch.ones_like(self.patch_weight)
                )

                self.featuresampler.set_sampling_weight(sampling_weight)
                self.patch_weight = self.patch_weight.clamp(min=0)

                filtered_indices = torch.where(sampling_weight > 0)[0]
                if len(filtered_indices) == 0:
                    LOGGER.warning("All features filtered - falling back to original")
                    filtered_indices = torch.arange(len(self.patch_weight))

                filtered_features = features[filtered_indices.cpu().numpy()]
                current_indices = current_indices[filtered_indices]
                current_scores = full_patch_weight[current_indices.cpu()].cpu().numpy()

                self.patch_weight = self.patch_weight[filtered_indices]
                sampling_weight = sampling_weight[filtered_indices]
                self.featuresampler.set_sampling_weight(sampling_weight)

                stage2_weight_np = current_scores.flatten() if len(current_scores) > 0 else np.array([0])
                stage2_skew = float(skew(stage2_weight_np))
                stage2_kurt = float(kurtosis(stage2_weight_np))
                LOGGER.info(f"[Stage 2: After anomaly removal] skewness={stage2_skew:.4f}, kurtosis={stage2_kurt:.4f}")

            # Stage 3: Redundancy removal (high similarity patches)
            if self.enable_normal_patch_removal:
                pre_patch_weight = self.patch_weight.clone().cpu().numpy().flatten()
                pre_skew = float(skew(pre_patch_weight))
                pre_kurt = float(kurtosis(pre_patch_weight))
                LOGGER.info(f"[Stage 3: Before redundancy removal] skewness={pre_skew:.4f}, kurtosis={pre_kurt:.4f} | patches={len(pre_patch_weight)}")

                filtered_features_tensor = torch.from_numpy(filtered_features).to(self.device)
                if filtered_features_tensor.ndim == 2:
                    filtered_features_tensor = filtered_features_tensor.unsqueeze(0)

                patch, batch, feature_dim = filtered_features_tensor.shape
                patch_features = filtered_features_tensor.view(-1, batch, feature_dim)

                patch_features_after, keep_mask = self.remove_redundant_patches_cosine_gpu(
                    patch_features, similarity_threshold=self.redundant_threshold
                )

                filtered_features = patch_features_after.cpu().numpy()
                retained_local_idx = keep_mask.nonzero(as_tuple=True)[0]
                current_indices = current_indices[retained_local_idx]

                # Update patch_weight safely
                if self.patch_weight is not None:
                    valid_retained = retained_local_idx[retained_local_idx < self.patch_weight.shape[0]]
                    if len(valid_retained) > 0:
                        self.patch_weight = self.patch_weight[valid_retained]
                        if 'sampling_weight' in locals():
                            sampling_weight = sampling_weight[valid_retained]
                            self.featuresampler.set_sampling_weight(sampling_weight)

                if self.patch_weight is not None and len(self.patch_weight) > 0:
                    post_patch_weight = self.patch_weight.cpu().numpy().flatten()
                    post_skew = float(skew(post_patch_weight))
                    post_kurt = float(kurtosis(post_patch_weight))
                    removed_count = len(pre_patch_weight) - len(post_patch_weight)
                    removed_ratio = removed_count / len(pre_patch_weight) * 100 if len(pre_patch_weight) > 0 else 0
                    LOGGER.info(f"[Stage 3: After redundancy removal] skewness={post_skew:.4f}, kurtosis={post_kurt:.4f}")
                    LOGGER.info(f"Removed {removed_count} patches ({removed_ratio:.2f}%) | threshold={self.redundant_threshold}")

            # Final coreset sampling (only once)
            sample_features, sample_indices = self.featuresampler.run(filtered_features)

            # Map indices safely and set coreset weights
            if isinstance(sample_indices, np.ndarray):
                sample_indices = torch.from_numpy(sample_indices).to(self.device).long()

            if not self.disable_patch_removal and self.patch_weight is not None:
                if len(sample_indices) > 0:
                    max_idx = sample_indices.max().item()
                    if max_idx < self.patch_weight.shape[0]:
                        self.coreset_weight = self.patch_weight[sample_indices.cpu().numpy()].cpu().numpy()
                    else:
                        LOGGER.warning("Coreset index out of bounds - fallback to uniform weights")
                        self.coreset_weight = np.ones(len(sample_features))
                else:
                    self.coreset_weight = np.ones(len(sample_features))
            else:
                self.coreset_weight = np.ones(len(sample_features))

            LOGGER.info(f"Coreset samples: {len(sample_features)} | coreset_weight size: {len(self.coreset_weight)}")

            self.anomaly_scorer.fit(detection_features=[sample_features])
            LOGGER.info(f"FAISS index size: {self.anomaly_scorer.nn_method.search_index.ntotal}")

    def _compute_patch_weight(self, features: np.ndarray):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        reduced_features = self.featuresampler._reduce_features(features)
        patch_features = reduced_features.reshape(
            -1, self.feature_shape[0] * self.feature_shape[1], reduced_features.shape[-1]
        )
        patch_features = patch_features.permute(1, 0, 2)

        if self.weight_method == "lof":
            patch_weight = self._compute_lof(self.lof_k, patch_features).transpose(-1, -2)
        elif self.weight_method == "lof_gpu":
            patch_weight = self._compute_lof_gpu(self.lof_k, patch_features).transpose(-1, -2)
        elif self.weight_method == "nearest":
            patch_weight = self._compute_nearest_distance(patch_features).transpose(-1, -2)
            patch_weight = patch_weight + 1
        elif self.weight_method == "gaussian":
            gaussian = multi_variate_gaussian.MultiVariateGaussian(patch_features.shape[2], patch_features.shape[0])
            stats = gaussian.fit(patch_features)
            patch_weight = self._compute_distance_with_gaussian(patch_features, stats).transpose(-1, -2)
            patch_weight = patch_weight + 1
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")

        return patch_weight

    def _compute_distance_with_gaussian(self, embedding: torch.Tensor, stats: list) -> torch.Tensor:
        embedding = embedding.permute(1, 2, 0)
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)
        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2)
        return torch.sqrt(distances)

    def _compute_nearest_distance(self, embedding: torch.Tensor) -> torch.Tensor:
        patch, batch, _ = embedding.shape
        x_x = (embedding ** 2).sum(dim=-1, keepdim=True).expand(patch, batch, batch)
        dist_mat = (x_x + x_x.transpose(-1, -2) - 2 * embedding.matmul(embedding.transpose(-1, -2))).abs() ** 0.5
        nearest_distance = torch.topk(dist_mat, dim=-1, largest=False, k=2)[0].sum(dim=-1)
        return nearest_distance

    def _compute_lof(self, k, embedding: torch.Tensor) -> torch.Tensor:
        patch, batch, _ = embedding.shape
        clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
        scores = torch.zeros(size=(patch, batch), device=embedding.device)
        for i in range(patch):
            clf.fit(embedding[i].cpu())
            scores[i] = torch.Tensor(-clf.negative_outlier_factor_)
        return scores

    def _compute_lof_gpu(self, k, embedding: torch.Tensor) -> torch.Tensor:
        patch, batch, _ = embedding.shape
        x_x = (embedding ** 2).sum(dim=-1, keepdim=True).expand(patch, batch, batch)
        dist_mat = (x_x + x_x.transpose(-1, -2) - 2 * embedding.matmul(embedding.transpose(-1, -2))).abs() ** 0.5 + 1e-6

        top_k_distance_mat, top_k_index = torch.topk(dist_mat, dim=-1, largest=False, k=k + 1)
        top_k_distance_mat, top_k_index = top_k_distance_mat[:, :, 1:], top_k_index[:, :, 1:]
        k_distance_value_mat = top_k_distance_mat[:, :, -1]

        reach_dist_mat = torch.max(dist_mat, k_distance_value_mat.unsqueeze(2).expand(patch, batch, batch).transpose(-1, -2))
        top_k_index_hot = torch.zeros(size=dist_mat.shape, device=top_k_index.device).scatter_(-1, top_k_index, 1)

        lrd_mat = k / (top_k_index_hot * reach_dist_mat).sum(dim=-1)
        lof_mat = ((lrd_mat.unsqueeze(2).expand(patch, batch, batch).transpose(-1, -2) * top_k_index_hot).sum(dim=-1) / k) / lrd_mat
        return lof_mat

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(dataloader, desc="Inferring...") as iterator:
            for batch in iterator:
                if isinstance(batch, dict):
                    labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                    masks_gt.extend(batch["mask"].numpy().tolist())
                    image = batch["image"]
                else:
                    image = batch
                _scores, _masks = self._predict(image)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer anomaly scores and segmentation masks for a batch."""
        if isinstance(images, dict):
            images = images["image"]
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            LOGGER.info(f"Testing patch shape: {patch_shapes[0]}")
            features = np.asarray(features)

            image_scores, _, indices = self.anomaly_scorer.predict([features])

            patch_scores = image_scores.copy()

            # Image-level score (max over patches)
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            # Patch-level scores for segmentation
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving model data.")
        self.anomaly_scorer.save(save_path, save_features_separately=False, prepend=prepend)
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(self, load_path: str, device: torch.device, nn_method=common.FaissNN(False, 4), prepend: str = ""):
        LOGGER.info("Loading model.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            params = pickle.load(load_file)
        params["backbone"] = backbones.load(params["backbone.name"])
        params["backbone"].name = params["backbone.name"]
        del params["backbone.name"]
        self.load(**params, device=device, nn_method=nn_method)
        self.anomaly_scorer.load(load_path, prepend)


class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride or patchsize

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for side in features.shape[-2:]:
            n_patches = (
                side + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, patch_scores, batchsize):
        return patch_scores.reshape(batchsize, -1, *patch_scores.shape[1:])

    def score(self, image_scores):
        was_numpy = isinstance(image_scores, np.ndarray)
        if was_numpy:
            image_scores = torch.from_numpy(image_scores)
        while image_scores.ndim > 1:
            image_scores = torch.max(image_scores, dim=-1).values
        return image_scores.numpy() if was_numpy else image_scores