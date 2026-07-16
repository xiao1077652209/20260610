"""
Fusion modules aligned with the MFFN-INIRS paper.
"""
import torch
import torch.nn as nn


SUPPORTED_FUSION_METHODS = ("dwgff", "concat", "acgf", "spectral_residual")
FUSION_ALIASES = {
    "adaptive_channel_gated": "acgf",
    "adaptive_channel_gated_fusion": "acgf",
    "channel_gated": "acgf",
    "light_gated": "acgf",
    "lightweight_gated": "acgf",
    "spectral_residual_gated": "spectral_residual",
    "srgf": "spectral_residual",
}


def normalize_fusion_name(name):
    normalized = str(name).lower()
    return FUSION_ALIASES.get(normalized, normalized)


class ConcatFusion(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        self.output_dim = feature_dim * 2

    def forward(self, xm, xv):
        return torch.cat([xm, xv], dim=1)


class DWGFF(nn.Module):
    """
    Paper definition:
    hm = tanh(Wm xm)
    hv = tanh(Wv xv)
    [w1, w2] = softmax(W [hm, hv])
    ffused = w1 * hm + w2 * hv
    """
    def __init__(self, feature_dim=1024, dropout=0.0):
        super().__init__()
        self.output_dim = feature_dim
        hidden_dim = max(128, feature_dim // 2)
        self.Wm = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.Wv = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, feature_dim, bias=True),
            nn.Sigmoid(),
        )
        self.post_fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, xm, xv):
        hm = self.Wm(xm)
        hv = self.Wv(xv)
        gate_input = torch.cat([hm, hv, torch.abs(hm - hv), hm * hv], dim=1)
        weights = self.gate(gate_input)
        fused = weights * hm + (1.0 - weights) * hv
        enhanced = torch.cat([fused, torch.abs(hm - hv), hm * hv], dim=1)
        return self.post_fusion(enhanced)


class AdaptiveChannelGatedFusion(nn.Module):
    """
    Conservative ACGF revision:
    - keep interaction and modality difference in gate generation
    - keep residual shortcut for information preservation
    - avoid over-heavy output concatenation to reduce overfitting
    """
    def __init__(self, feature_dim=1024, hidden_dim=256, dropout=0.0):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.output_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.image_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
        )
        self.spectral_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim * 3, feature_dim, bias=True),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, xm, xv):
        hm = self.image_proj(xm)
        hv = self.spectral_proj(xv)

        interaction = hm * hv
        difference = torch.abs(hm - hv)
        image_gate_input = torch.cat([hm, interaction, difference], dim=1)
        spectral_gate_input = torch.cat([hv, interaction, difference], dim=1)

        image_gate = self.image_gate(image_gate_input)
        spectral_gate = self.spectral_gate(spectral_gate_input)

        fused_image = image_gate * hm + (1.0 - image_gate) * interaction
        fused_spectral = spectral_gate * hv + (1.0 - spectral_gate) * interaction

        fused = torch.cat([fused_image, fused_spectral], dim=1)
        shortcut = torch.cat([hm, hv], dim=1)
        refined = self.refine(fused) + shortcut
        merged = torch.cat([refined, difference], dim=1)
        return self.compress(merged)


class SpectralResidualFusion(nn.Module):
    """Preserve the spectral representation and add a gated secondary-view residual."""
    def __init__(self, feature_dim=1024, hidden_dim=256, dropout=0.0, initial_secondary_weight=0.05):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if not 0.0 < initial_secondary_weight < 1.0:
            raise ValueError("initial_secondary_weight must be between 0 and 1.")
        self.output_dim = feature_dim
        self.secondary_adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
        )
        gate_hidden = max(64, hidden_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, feature_dim),
            nn.Sigmoid(),
        )
        initial_logit = torch.logit(torch.tensor(float(initial_secondary_weight)))
        self.secondary_scale_logit = nn.Parameter(initial_logit)

    def forward(self, secondary_features, spectral_features):
        secondary_delta = self.secondary_adapter(secondary_features)
        gate_input = torch.cat([
            secondary_features,
            spectral_features,
            torch.abs(secondary_features - spectral_features),
        ], dim=1)
        gated_delta = self.gate(gate_input) * secondary_delta
        secondary_scale = torch.sigmoid(self.secondary_scale_logit)
        return spectral_features + secondary_scale * gated_delta


def build_fusion(name="dwgff", feature_dim=1024, dropout=0.0, hidden_dim=None, initial_secondary_weight=0.05):
    name = normalize_fusion_name(name)
    if name == "dwgff":
        return DWGFF(feature_dim=feature_dim, dropout=dropout)
    if name == "concat":
        return ConcatFusion(feature_dim=feature_dim)
    if name == "acgf":
        if hidden_dim is None:
            hidden_dim = min(256, feature_dim)
        return AdaptiveChannelGatedFusion(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if name == "spectral_residual":
        if hidden_dim is None:
            hidden_dim = min(256, feature_dim)
        return SpectralResidualFusion(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            initial_secondary_weight=initial_secondary_weight,
        )
    raise ValueError(f"Unsupported fusion method: {name}")
