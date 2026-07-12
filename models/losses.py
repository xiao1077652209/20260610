import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Class-compactness loss applied on fused features.

    This is adapted from the reference spectral branch idea, but implemented in
    a lighter form that is easier to plug into the current training pipeline.
    """

    def __init__(self, num_classes, feat_dim, normalize=True, eps=1e-6):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if feat_dim < 1:
            raise ValueError(f"feat_dim must be >= 1, got {feat_dim}")
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim) * 0.02)

    def forward(self, features, labels):
        if features.ndim != 2:
            raise ValueError(f"features must be 2D [batch, feat_dim], got shape {tuple(features.shape)}")
        if features.size(1) != self.feat_dim:
            raise ValueError(
                f"Expected feature dim {self.feat_dim}, got {features.size(1)}"
            )

        labels = labels.view(-1).long()
        if features.size(0) != labels.size(0):
            raise ValueError(
                f"Batch mismatch: features={features.size(0)}, labels={labels.size(0)}"
            )

        centers = self.centers
        if self.normalize:
            features = F.normalize(features, p=2, dim=1, eps=self.eps)
            centers = F.normalize(centers, p=2, dim=1, eps=self.eps)

        batch_centers = centers.index_select(0, labels)
        return (features - batch_centers).pow(2).sum(dim=1).mean()
