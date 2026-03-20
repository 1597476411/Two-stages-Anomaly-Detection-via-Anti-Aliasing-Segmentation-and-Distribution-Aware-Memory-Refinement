import timm
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union, List, Optional
import os

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=False)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b4": 'timm.create_model("tf_efficientnet_b4", pretrained=False)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
    
    # DINO models
    "DINO_VIT-S/8": "timm.create_model('vit_small_patch8_224', pretrained=False)",
    "DINO_VIT-B/16": "timm.create_model('vit_base_patch8_224', pretrained=False)",
    
    "mobilenetV2": "models.mobilenet_v2(pretrained=True)",
    "mobilenetV3_large": "models.mobilenet_v3_large(pretrained=True)",
    "mobilenetV3_small": "models.mobilenet_v3_small(pretrained=True)",
    
    "resnext50_32x4d": "models.resnext50_32x4d(pretrained=True)",
}

class BackboneWrapper(nn.Module):
    def __init__(self, 
                 name: str = "moblidenetV2",
                 pretrained: bool = True,
                 features_only: bool = False,
                 out_indices: Optional[List[int]] = None,
                 seed: Optional[int] = None,
                 ):
        super().__init__()
        self.name = name
        self.features_only = features_only
        self.out_indices = out_indices
        self.model = None
        self.seed = seed

        try:
            self.model = eval(_BACKBONES[name])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model for {name}: {str(e)}")

        for module_name, module in self.model.named_children():
            self.add_module(module_name, module)

        if features_only:
            if 'wide' in name:
                self.feature_layers = [self.model.layer2, self.model.layer3]
            else:
                self.feature_layers = [self.model.layer3]

    
    
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self.features_only:
            return self.model(x)
        features = []
        for layer in self.feature_layers:
            x = layer(x)
            features.append(x)
        return features
    
    @property
    def feature_dim(self) -> int:
        """Get the output feature dimension"""
        if 'wide' in self.name:
            return 1024 if '50' in self.name else 2048
        elif 'resnet' in self.name:
            return 2048
        else:
            return 512  # Default for other backbones



def load_backbone(name: str, pretrained_path: str = None, pretrained: bool = True, seed: Optional[int] = None, **kwargs) -> BackboneWrapper:
    if name not in _BACKBONES:
        raise ValueError(f"Unsupported backbone: {name}. Available: {list(_BACKBONES.keys())}")    
    wrapper = BackboneWrapper(
        name=name,
        pretrained=pretrained,
        seed=seed,
        **kwargs
    )
    
    if not hasattr(wrapper, 'model') or wrapper.model is None:
        raise RuntimeError(f"Model not initialized for {name}")
    
    return wrapper


# Compatibility with original load function
load = load_backbone