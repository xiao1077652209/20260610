"""
Image backbone factories aligned with the MFFN-INIRS paper.
"""
import torch
import torch.nn as nn
import torchvision.models as models


SUPPORTED_IMAGE_BACKBONES = (
    "resnet50",
    "mobilenet_v2",
    "alexnet",
    "vgg16",
    "shufflenet_v2",
)

IMAGE_BACKBONE_ALIASES = {
    "mobilenetv2": "mobilenet_v2",
    "shufflenet": "shufflenet_v2",
    "shufflenet_v2_x1_0": "shufflenet_v2",
}


def normalize_image_backbone_name(name):
    normalized = str(name).lower()
    return IMAGE_BACKBONE_ALIASES.get(normalized, normalized)


def _load_model(model_fn, weights_enum, pretrained, name):
    weights = weights_enum.IMAGENET1K_V1 if pretrained else None
    try:
        return model_fn(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                f"Failed to load pretrained weights for {name}. "
                "Set IMAGE_PRETRAINED = False only if you intentionally "
                "want random initialization."
            ) from exc
        raise


def _adapt_first_conv(module, in_channels):
    if in_channels == 3:
        return

    conv = module
    weight = conv.weight.detach()
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_weight = weight.mean(dim=1, keepdim=True)
        elif in_channels > 3:
            repeat = (in_channels + 2) // 3
            expanded = weight.repeat(1, repeat, 1, 1)[:, :in_channels]
            new_weight = expanded * (3.0 / float(in_channels))
        else:
            new_weight = weight[:, :in_channels] * (3.0 / float(in_channels))
        new_conv.weight.copy_(new_weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.detach())
    return new_conv


def _freeze_resnet_stages(backbone, freeze_stages):
    stage_modules = [backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
    for module in stage_modules[:max(0, freeze_stages + 1)]:
        for param in module.parameters():
            param.requires_grad = False


def _freeze_mobilenet_stages(backbone, freeze_stages):
    features = list(backbone.features.children())
    cutoff = min(len(features), max(0, freeze_stages) * 2 + 2)
    for module in features[:cutoff]:
        for param in module.parameters():
            param.requires_grad = False


def _freeze_vgg_or_alexnet_stages(feature_extractor, freeze_stages):
    layers = list(feature_extractor.children())
    cutoff = min(len(layers), max(0, freeze_stages) * 3 + 2)
    for module in layers[:cutoff]:
        for param in module.parameters():
            param.requires_grad = False


def _freeze_shufflenet_stages(backbone, freeze_stages):
    stage_modules = [backbone.conv1, backbone.maxpool, backbone.stage2, backbone.stage3, backbone.stage4]
    for module in stage_modules[:min(len(stage_modules), max(0, freeze_stages) + 2)]:
        for param in module.parameters():
            param.requires_grad = False


def freeze_backbone_stages(model, backbone_name, freeze_stages):
    if freeze_stages <= 0:
        return
    if backbone_name == "resnet50":
        _freeze_resnet_stages(model, freeze_stages)
    elif backbone_name == "mobilenet_v2":
        _freeze_mobilenet_stages(model, freeze_stages)
    elif backbone_name in {"alexnet", "vgg16"}:
        _freeze_vgg_or_alexnet_stages(model.features, freeze_stages)
    elif backbone_name == "shufflenet_v2":
        _freeze_shufflenet_stages(model, freeze_stages)


class _PaperImageBackbone(nn.Module):
    def __init__(self, backbone_name, output_dim=1024, pretrained=True, in_channels=3, freeze_stages=0, dropout=0.0):
        super().__init__()
        backbone_name = normalize_image_backbone_name(backbone_name)
        self.backbone_name = backbone_name

        if backbone_name == "resnet50":
            backbone = _load_model(models.resnet50, models.ResNet50_Weights, pretrained, backbone_name)
            if in_channels != 3:
                backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels)
            freeze_backbone_stages(backbone, backbone_name, freeze_stages)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_head = nn.Identity()
            in_features = 2048
        elif backbone_name == "mobilenet_v2":
            backbone = _load_model(models.mobilenet_v2, models.MobileNet_V2_Weights, pretrained, backbone_name)
            if in_channels != 3:
                backbone.features[0][0] = _adapt_first_conv(backbone.features[0][0], in_channels)
            freeze_backbone_stages(backbone, backbone_name, freeze_stages)
            self.feature_extractor = backbone.features
            self.feature_head = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 1280
        elif backbone_name == "alexnet":
            backbone = _load_model(models.alexnet, models.AlexNet_Weights, pretrained, backbone_name)
            if in_channels != 3:
                backbone.features[0] = _adapt_first_conv(backbone.features[0], in_channels)
            freeze_backbone_stages(backbone, backbone_name, freeze_stages)
            self.feature_extractor = backbone.features
            self.feature_head = nn.Sequential(
                backbone.avgpool,
                nn.Flatten(),
                nn.Sequential(*list(backbone.classifier.children())[:-1]),
            )
            in_features = 4096
        elif backbone_name == "vgg16":
            backbone = _load_model(models.vgg16, models.VGG16_Weights, pretrained, backbone_name)
            if in_channels != 3:
                backbone.features[0] = _adapt_first_conv(backbone.features[0], in_channels)
            freeze_backbone_stages(backbone, backbone_name, freeze_stages)
            self.feature_extractor = backbone.features
            self.feature_head = nn.Sequential(
                backbone.avgpool,
                nn.Flatten(),
                nn.Sequential(*list(backbone.classifier.children())[:-1]),
            )
            in_features = 4096
        elif backbone_name == "shufflenet_v2":
            backbone = _load_model(
                models.shufflenet_v2_x1_0,
                models.ShuffleNet_V2_X1_0_Weights,
                pretrained,
                backbone_name,
            )
            if in_channels != 3:
                backbone.conv1[0] = _adapt_first_conv(backbone.conv1[0], in_channels)
            freeze_backbone_stages(backbone, backbone_name, freeze_stages)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_head = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 1024
        else:
            raise ValueError(f"Unsupported image backbone: {backbone_name}")

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, output_dim),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.feature_head(x)
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


def build_image_backbone(name="resnet50", output_dim=1024, pretrained=True, in_channels=3, freeze_stages=0, dropout=0.0):
    return _PaperImageBackbone(
        name,
        output_dim=output_dim,
        pretrained=pretrained,
        in_channels=in_channels,
        freeze_stages=freeze_stages,
        dropout=dropout,
    )
