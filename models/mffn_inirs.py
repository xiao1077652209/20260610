"""
Paper-style MFFN-INIRS network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import build_fusion, normalize_fusion_name
from models.image_backbones import build_image_backbone, normalize_image_backbone_name
from models.spectral_extractors import build_spectral_extractor_with_channels, normalize_spectral_backbone_name


class MFFNINIRS(nn.Module):
    def __init__(
        self,
        spectral_length,
        num_classes,
        feature_dim=1024,
        image_backbone="resnet50",
        spectral_backbone="cnn1d",
        fusion_method="dwgff",
        image_pretrained=True,
        image_in_channels=3,
        spectral_in_channels=1,
        freeze_image_backbone_stages=0,
        image_dropout=0.0,
        fusion_dropout=0.0,
        fusion_hidden_dim=None,
        fusion_initial_image_weight=0.05,
        modality_mode="multimodal",
    ):
        super().__init__()
        self.modality_mode = str(modality_mode).lower()
        if self.modality_mode not in ("multimodal", "spectral_only", "image_only"):
            raise ValueError(f"Unsupported modality mode: {modality_mode}")
        self.image_backbone_name = normalize_image_backbone_name(image_backbone)
        self.spectral_backbone_name = normalize_spectral_backbone_name(spectral_backbone)
        self.fusion_name = normalize_fusion_name(fusion_method)

        self.image_branch = build_image_backbone(
            self.image_backbone_name,
            output_dim=feature_dim,
            pretrained=image_pretrained,
            in_channels=image_in_channels,
            freeze_stages=freeze_image_backbone_stages,
            dropout=image_dropout,
        ) if self.modality_mode != "spectral_only" else None
        self.spectral_branch = build_spectral_extractor_with_channels(
            self.spectral_backbone_name,
            input_length=spectral_length,
            output_dim=feature_dim,
            in_channels=spectral_in_channels,
        ) if self.modality_mode != "image_only" else None
        self.fusion = build_fusion(
            self.fusion_name,
            feature_dim=feature_dim,
            dropout=fusion_dropout,
            hidden_dim=fusion_hidden_dim,
            initial_image_weight=fusion_initial_image_weight,
        )
        self.fused_dim = feature_dim if self.modality_mode != "multimodal" else self.fusion.output_dim
        bottleneck_dim = max(128, self.fused_dim // 2)
        self.fusion_norm = nn.LayerNorm(self.fused_dim)
        self.feature_head = nn.Sequential(
            nn.Linear(self.fused_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=max(float(image_dropout), float(fusion_dropout), 0.15)),
        )
        self.embedding_dim = bottleneck_dim
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.image_aux_classifier = nn.Linear(feature_dim, num_classes)
        self.spectral_aux_classifier = nn.Linear(feature_dim, num_classes)

    def _encode_modalities(self, image_input, spectral_input):
        xm = None if self.image_branch is None else F.normalize(self.image_branch(image_input), p=2, dim=1)
        xv = None if self.spectral_branch is None else F.normalize(self.spectral_branch(spectral_input), p=2, dim=1)
        return xm, xv

    def extract_features(self, image_input, spectral_input):
        with torch.no_grad():
            outputs = self.forward(image_input, spectral_input, return_dict=True)
        return outputs["fused_features"]

    def forward(self, image_input, spectral_input, return_features=False, return_dict=False):
        xm, xv = self._encode_modalities(image_input, spectral_input)
        if self.modality_mode == "image_only":
            fused = xm
        elif self.modality_mode == "spectral_only":
            fused = xv
        else:
            fused = self.fusion(xm, xv)
        fused = self.fusion_norm(fused)
        embedding = self.feature_head(fused)
        logits = self.classifier(embedding)
        image_logits = None if xm is None else self.image_aux_classifier(xm)
        spectral_logits = None if xv is None else self.spectral_aux_classifier(xv)
        if return_dict:
            return {
                "logits": logits,
                "features": fused,
                "embedding": embedding,
                "fused_features": fused,
                "image_features": xm,
                "spectral_features": xv,
                "image_logits": image_logits,
                "spectral_logits": spectral_logits,
            }
        if return_features:
            return fused
        return logits

    def freeze_spectral_backbone(self):
        if self.spectral_branch is None:
            raise RuntimeError("Cannot freeze a missing spectral branch.")
        for parameter in self.spectral_branch.parameters():
            parameter.requires_grad = False
        self.spectral_branch.eval()

    def keep_frozen_modules_in_eval(self):
        if self.spectral_branch is not None and not any(
            parameter.requires_grad for parameter in self.spectral_branch.parameters()
        ):
            self.spectral_branch.eval()
