# 模型架构说明

## 1. 总体架构

本项目主模型是 `MFFNINIRS`，定义在 `models/mffn_inirs.py`。它是一个面向近红外光谱分类的双分支多模态融合架构：

```text
原始光谱 CSV
    |
    |-- 光谱预处理 -> 一维光谱输入 -> 光谱分支 -> 光谱特征 xv
    |
    |-- 光谱图像编码 -> 二维图像输入 -> 图像分支 -> 图像特征 xm

xm + xv -> ACGF/DWGFF/Concat 融合 -> 融合特征 fused
        -> 深度分类头训练
        -> 提取特征
        -> 传统分类器或 FC 分类头输出最终结果
```

当前默认配置为：

```python
ENCODING_METHOD = "rp"
IMAGE_BACKBONE = "resnet50"
SPECTRAL_BACKBONE = "attn_cnn"
FUSION_METHOD = "acgf"
FEATURE_DIM = 512
FINAL_CLASSIFIER = "linear_svm"
FINAL_CLASSIFIER_FEATURE_MODE = "hybrid"
```

## 2. 数据输入与预处理

CSV 第一列是类别标签，其余列是光谱变量。程序读取数据后，会将标签编码为从 `0` 开始的整数类别。

默认光谱目标长度为：

```python
SPECTRAL_TARGET_LENGTH = 778
ENCODING_TARGET_LENGTH = 778
```

如果原始光谱长度不同，先用线性插值重采样到目标长度。默认光谱预处理为：

```python
SPECTRAL_PREPROCESSING_METHOD = "msc+sg+d1"
```

即：

```text
MSC 多元散射校正 -> SG 平滑 -> 一阶差分
```

注意：`d1` 一阶差分会使一维光谱分支实际输入长度从 `778` 变为 `777`。二维图像编码仍会按 `ENCODING_TARGET_LENGTH = 778` 生成图像输入。

## 3. 二维图像模态构造

项目会将每条一维光谱转换为二维图像。支持的编码方式包括：

```text
rp, gasf, gadf, mtf
```

当前默认是 `rp`，即 Recurrence Plot。编码后的矩阵会缓存为 `.npy` 文件。`SpectralImageDataset` 读取缓存后会：

```text
二维矩阵 -> clip/scale -> uint8 -> 复制为 RGB 三通道 -> Resize(224, 224) -> ImageNet 归一化
```

所以图像分支输入形状为：

```text
[batch, 3, 224, 224]
```

一维光谱分支输入形状为：

```text
[batch, 1, spectral_length]
```

## 4. 图像分支

图像分支由 `models/image_backbones.py` 构建。支持：

```text
resnet50, mobilenet_v2, alexnet, vgg16, shufflenet_v2
```

默认使用 `resnet50`，并加载 ImageNet 预训练权重：

```python
IMAGE_PRETRAINED = True
FREEZE_IMAGE_BACKBONE_STAGES = 2
IMAGE_DROPOUT = 0.10
```

默认 ResNet50 图像分支流程为：

```text
RGB 光谱编码图像
-> ResNet50 去除最终分类层
-> 2048 维视觉特征
-> Dropout
-> Linear(2048 -> FEATURE_DIM)
-> L2 normalize
```

默认输出：

```text
xm: [batch, 512]
```

## 5. 光谱分支

光谱分支由 `models/spectral_extractors.py` 构建。支持：

```text
cnn1d, attn_cnn, lstm, bilstm
```

当前默认是 `attn_cnn`。其结构为：

```text
Conv1d(1 -> 128, kernel=7)
-> BN -> GELU -> AvgPool
-> 光谱注意力
-> Conv1d(128 -> 256, kernel=5)
-> BN -> GELU -> AvgPool
-> 光谱注意力
-> Conv1d(256 -> 128, kernel=3)
-> BN -> GELU -> AvgPool
-> 光谱注意力
-> AdaptiveAvgPool1d + AdaptiveMaxPool1d
-> 拼接池化特征
-> Linear + LayerNorm + GELU + Dropout
-> Linear 到 FEATURE_DIM
```

默认输出：

```text
xv: [batch, 512]
```

`MFFNINIRS` 在融合前还会对 `xm` 和 `xv` 再做一次 L2 归一化，使两个模态的特征尺度更接近。

## 6. 融合模块

融合模块由 `models/fusion.py` 构建。支持：

```text
concat, dwgff, acgf
```

当前默认是 `acgf`，即自适应通道门控融合。默认隐藏维度：

```python
FUSION_HIDDEN_DIM = 256
```

ACGF 处理流程：

```text
xm -> image_proj    -> hm: [batch, 256]
xv -> spectral_proj -> hv: [batch, 256]

interaction = hm * hv
difference  = abs(hm - hv)

image_gate    = gate([hm, interaction, difference])
spectral_gate = gate([hv, interaction, difference])

fused_image    = image_gate * hm + (1 - image_gate) * interaction
fused_spectral = spectral_gate * hv + (1 - spectral_gate) * interaction

refined = refine([fused_image, fused_spectral]) + [hm, hv]
merged  = [refined, difference]
fused   = compress(merged) -> [batch, 512]
```

该模块显式建模：

- 图像模态特征。
- 光谱模态特征。
- 两模态乘积交互。
- 两模态绝对差异。
- 通道级门控权重。
- 残差信息保留。

## 7. 深度分类头

融合特征进入 `MFFNINIRS` 的分类头：

```text
fused: [batch, 512]
-> LayerNorm(512)
-> Linear(512 -> 256)
-> LayerNorm(256)
-> GELU
-> Dropout
-> Linear(256 -> num_classes)
```

输出：

```text
logits: [batch, num_classes]
```

模型同时输出：

```text
fused_features    融合特征
image_features    图像分支特征
spectral_features 光谱分支特征
embedding         分类头前的瓶颈特征
image_logits      图像分支辅助分类输出
spectral_logits   光谱分支辅助分类输出
```

## 8. 训练目标

主分类损失为带标签平滑的交叉熵：

```python
CrossEntropyLoss(label_smoothing=0.05)
```

默认启用分支辅助分类损失：

```python
USE_BRANCH_AUX_LOSS = True
IMAGE_AUX_LOSS_WEIGHT = 0.05
SPECTRAL_AUX_LOSS_WEIGHT = 0.10
```

因此默认训练目标为：

```text
total_loss =
  主分类交叉熵
  + 0.05 * 图像分支交叉熵
  + 0.10 * 光谱分支交叉熵
```

项目还支持两个可选损失：

```python
USE_CENTER_LOSS = False
USE_MODAL_ALIGN_LOSS = False
```

- `CenterLoss`：约束融合特征的类内紧凑性。
- `ModalAlignLoss`：约束图像特征和光谱特征的余弦一致性。

当前默认二者均关闭。

## 9. 训练策略

默认训练配置：

```python
OPTIMIZER = AdamW
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "plateau"
BATCH_SIZE = 64
EPOCHS = 60
EARLY_STOPPING_ENABLED = True
```

每轮训练后在验证集上计算损失和准确率。最佳模型根据：

```python
BEST_MODEL_METRIC = "val_acc_or_loss"
```

保存，即优先关注验证准确率，同时允许验证损失明显变好时更新最佳权重。

## 10. 最终分类器

深度模型训练完成后，项目会提取训练集、验证集和测试集的深度特征，再训练最终分类器。

当前默认特征组合方式：

```python
FINAL_CLASSIFIER_FEATURE_MODE = "hybrid"
```

`hybrid` 会拼接：

```text
[fused_features, image_features, spectral_features, abs(image_features - spectral_features)]
```

默认每块特征为 512 维，因此最终分类器输入为：

```text
512 + 512 + 512 + 512 = 2048 维
```

之后执行特征选择：

```python
CLASSIFIER_USE_FEATURE_SELECTION = True
CLASSIFIER_MAX_FEATURES = 512
```

最终默认分类器是：

```python
FINAL_CLASSIFIER = "linear_svm"
```

程序会在多个 `C` 值之间做交叉验证，选择表现最好的线性 SVM，再在测试集上输出 Accuracy、Precision、Recall、F1-score、混淆矩阵和 ROC 曲线。

## 11. 架构核查结论

当前主架构的张量流是闭合的：

```text
图像分支输出: [batch, 512]
光谱分支输出: [batch, 512]
ACGF 融合输出: [batch, 512]
瓶颈 embedding: [batch, 256]
分类 logits: [batch, num_classes]
hybrid 最终分类特征: [batch, 2048]
```

已修正的架构问题：

- `USE_MODAL_ALIGN_LOSS` 和 `MODAL_ALIGN_LOSS_WEIGHT` 原本是死配置，启用后不会进入训练目标。
- 现已补充模态对齐损失、参数校验、训练日志、实验命名和结果汇总字段。

当前默认配置下模态对齐损失仍关闭，因此不会改变既有默认实验结果。
