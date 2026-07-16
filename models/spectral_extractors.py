"""
Spectral feature extractors aligned with the MFFN-INIRS paper.
"""
import torch
import torch.nn as nn


SUPPORTED_SPECTRAL_BACKBONES = ("cnn1d", "attn_cnn", "multiscale_cnn", "lstm", "bilstm")
SPECTRAL_BACKBONE_ALIASES = {
    "1d-cnn": "cnn1d",
    "attention_cnn": "attn_cnn",
    "attention-cnn": "attn_cnn",
    "enhanced_cnn": "attn_cnn",
    "enhancedcnn": "attn_cnn",
    "reference_cnn": "multiscale_cnn",
    "multiscale": "multiscale_cnn",
    "bi_lstm": "bilstm",
    "bidirectional_lstm": "bilstm",
}


def normalize_spectral_backbone_name(name):
    normalized = str(name).lower()
    return SPECTRAL_BACKBONE_ALIASES.get(normalized, normalized)


class PaperCNN1D(nn.Module):
    """
    Paper-style 1D-CNN:
    3 x (Conv1d + BN + ReLU + MaxPool1d), then Dropout(0.5), Flatten, FC.
    """
    def __init__(self, input_length, output_dim=1024, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout(p=0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, input_length)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            flattened_dim = dummy.flatten(1).shape[1]

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flattened_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.fc(x)


class SpectralAttention(nn.Module):
    """
    Channel and wavelength-position attention adapted from the reference spectral branch.
    """
    def __init__(self, in_channels, channel_att_strength=1.0, reduction=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.channel_att_strength = channel_att_strength
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.pos_att = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_weight = self.channel_att(x) ** self.channel_att_strength
        position_weight = self.pos_att(x)
        return x * channel_weight * position_weight


def _attention_conv_block(in_channels, out_channels, kernel_size, channel_att_strength, dropout):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
        nn.AvgPool1d(kernel_size=2, stride=2),
        SpectralAttention(out_channels, channel_att_strength=channel_att_strength),
        nn.Dropout(p=dropout),
    )


class AttentionCNN1D(nn.Module):
    """
    Attention-enhanced spectral branch for small NIR datasets.
    """
    def __init__(self, input_length, output_dim=1024, in_channels=1):
        super().__init__()
        # input_length is unused — model supports variable-length input
        self.blocks = nn.Sequential(
            _attention_conv_block(in_channels, 128, kernel_size=7, channel_att_strength=0.5, dropout=0.10),
            _attention_conv_block(128, 256, kernel_size=5, channel_att_strength=1.0, dropout=0.15),
            _attention_conv_block(256, 128, kernel_size=3, channel_att_strength=1.0, dropout=0.20),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        pooled_dim = 128 * 2
        hidden_dim = min(512, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.blocks(x)
        avg_feat = self.avgpool(x).flatten(1)
        max_feat = self.maxpool(x).flatten(1)
        pooled = torch.cat([avg_feat, max_feat], dim=1)
        return self.projection(pooled)


class ResidualSpectralBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            SpectralAttention(channels, channel_att_strength=0.5),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class MultiScaleSpectralCNN(nn.Module):
    """Peak-preserving NIR encoder inspired by the reference end-to-end branch.
    
    Reduced capacity version for small datasets (e.g., 产地 with 535 samples).
    - branch_channels: 48 -> 32
    - encoder channels: 192 -> 128
    - projection dim: 512 -> 256
    - increased dropout in residual blocks for regularization
    """
    def __init__(self, input_length, output_dim=1024, in_channels=1):
        super().__init__()
        # input_length is unused — model supports variable-length input
        branch_channels = 32  # 从 48 改为 32，降低模型容量
        self.multiscale_stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
            )
            for k in (3, 7, 15)
        ])
        channels = branch_channels * 3
        self.merge = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1, bias=False),  # 从 192 改为 128
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ResidualSpectralBlock(128, kernel_size=7, dilation=1, dropout=0.15),  # 从 0.08 改为 0.15
            nn.AvgPool1d(2),
            ResidualSpectralBlock(128, kernel_size=5, dilation=2, dropout=0.20),  # 从 0.10 改为 0.20
            nn.AvgPool1d(2),
            ResidualSpectralBlock(128, kernel_size=3, dilation=4, dropout=0.25),  # 从 0.12 改为 0.25
        )
        # Several bins retain coarse wavelength position, unlike global pooling alone.
        self.position_pool = nn.AdaptiveAvgPool1d(8)
        self.peak_pool = nn.AdaptiveMaxPool1d(8)
        self.projection = nn.Sequential(
            nn.Linear(128 * 16, 256),  # 从 512 改为 256
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),  # 从 0.25 改为 0.30
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.multiscale_stem], dim=1)
        x = self.encoder(self.merge(x))
        pooled = torch.cat([self.position_pool(x), self.peak_pool(x)], dim=1)
        return self.projection(pooled.flatten(1))


class LSTMExtractor(nn.Module):
    def __init__(self, output_dim=1024, bidirectional=False, input_channels=1):
        super().__init__()
        hidden_size = 128
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=bidirectional,
        )
        in_features = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        return self.fc(last_hidden)


def build_spectral_extractor(name="cnn1d", input_length=401, output_dim=1024):
    return build_spectral_extractor_with_channels(name=name, input_length=input_length, output_dim=output_dim, in_channels=1)


def build_spectral_extractor_with_channels(name="cnn1d", input_length=401, output_dim=1024, in_channels=1):
    name = normalize_spectral_backbone_name(name)
    if name == "cnn1d":
        return PaperCNN1D(input_length=input_length, output_dim=output_dim, in_channels=in_channels)
    if name == "attn_cnn":
        return AttentionCNN1D(input_length=input_length, output_dim=output_dim, in_channels=in_channels)
    if name == "multiscale_cnn":
        return MultiScaleSpectralCNN(input_length=input_length, output_dim=output_dim, in_channels=in_channels)
    if name == "lstm":
        return LSTMExtractor(output_dim=output_dim, bidirectional=False, input_channels=in_channels)
    if name == "bilstm":
        return LSTMExtractor(output_dim=output_dim, bidirectional=True, input_channels=in_channels)
    raise ValueError(f"Unsupported spectral extractor: {name}")
