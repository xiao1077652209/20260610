# 项目说明文档

## 1. 项目概述

本项目是一个面向近红外光谱数据分类的多模态深度学习实验工程。代码主线复现并扩展了 MFFN-INIRS 思路：将一维光谱序列转换为二维图像表示，同时保留原始或预处理后的一维光谱分支，通过多模态特征融合网络提取融合特征，最后使用神经网络分类头或传统机器学习分类器完成类别识别。

当前默认配置使用 `dataset/20251203_535_产地_7.csv` 作为数据源，第一列为类别标签，后续列为光谱变量。程序会自动检测 CSV 是否包含表头，并可将光谱长度重采样到统一长度。

## 2. 代码结构

```text
.
├── main.py                    # 项目入口，调用 paper_pipeline.main()
├── paper_pipeline.py          # 主流程：数据读取、划分、编码、训练、评估、结果保存
├── mffn_config.py             # 全局配置：数据路径、模型结构、训练参数、实验开关
├── validate_methods.py        # 方法组合烟雾测试入口
├── process.py                 # 光谱预处理方法集合
├── multimodal.py              # 早期/备用多模态模型文件，当前主流程未使用
├── models/
│   ├── mffn_inirs.py          # MFFN-INIRS 主模型
│   ├── image_backbones.py     # 图像分支骨干网络
│   ├── spectral_extractors.py # 光谱分支骨干网络
│   ├── fusion.py              # 特征融合模块
│   └── losses.py              # Center Loss
├── utils/
│   ├── image_encoding.py      # 光谱转图像编码：RP/GASF/GADF/MTF
│   ├── oversampling.py        # KMeans-SMOTE 过采样
│   └── pso_svm.py             # PSO-SVM 分类器
├── dataset/                   # 数据文件
└── results/                   # 训练结果、曲线图、混淆矩阵、ROC 等输出
```

## 3. 核心流程

运行 `python main.py` 后，程序会按以下流程执行：

1. 读取 `mffn_config.py` 中的配置并校验参数。
2. 从 CSV 读取数据，第一列编码为类别标签，其余列作为光谱特征。
3. 若光谱长度与 `SPECTRAL_TARGET_LENGTH` 不一致，则通过线性插值重采样。
4. 根据 `EVALUATION_PROTOCOL` 选择评估协议：
   - `paper`：先对全数据执行 KMeans-SMOTE，再按 7:2:1 划分训练集、验证集、测试集。
   - `strict`：先划分数据，再仅对训练集过采样，更适合避免数据泄漏。
5. 对光谱执行预处理，例如当前默认的 `msc+sg+d1`。
6. 将光谱序列编码为二维图像，并缓存到 `encodings/`。
7. 构建 MFFN-INIRS 模型：
   - 图像分支提取二维编码图像特征。
   - 光谱分支提取一维光谱特征。
   - 融合模块合并两路特征。
   - 分类头输出深度模型预测。
8. 使用训练集训练模型，并在验证集上选择最佳权重。
9. 提取融合特征、图像特征、光谱特征，按配置组合为最终分类器输入。
10. 使用最终分类器在测试集上评估，并输出 Accuracy、Precision、Recall、F1-score、混淆矩阵和 ROC 曲线。

## 4. 数据格式

CSV 数据应满足：

```text
label, band_1, band_2, band_3, ...
0,     0.3629, 0.3629, 0.3630, ...
1,     0.4120, 0.4118, 0.4116, ...
```

要求：

- 第一列必须是类别标签，可为数字或可排序的标签值。
- 第二列开始必须是可转换为数值的光谱特征。
- `CSV_HAS_HEADER = "auto"` 时，程序自动判断首行是否为表头。
- 当前默认数据文件共有 535 行、1556 列，即 1 列标签和 1555 个光谱变量；主流程会重采样到 778 个波段。

## 5. 主要配置说明

所有核心实验参数集中在 `mffn_config.py`。

### 数据与路径

```python
DATA_PATH = os.path.join(DATASET_DIR, "20251203_535_产地_7.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODING_ROOT = os.path.join(BASE_DIR, "encodings")
```

- `DATA_PATH`：单文件数据集路径。
- `RESULTS_DIR`：模型、指标、图像结果输出目录。
- `ENCODING_ROOT`：二维图像编码缓存目录。

### 光谱长度与预处理

```python
SPECTRAL_TARGET_LENGTH = 778
ENCODING_TARGET_LENGTH = 778
SPECTRAL_PREPROCESSING_METHOD = "msc+sg+d1"
```

支持的单步预处理方法包括：

```text
none, mms, ss, ct, snv, ma, sg, msc, d1, d2, dt, wave
```

可以用 `+` 串联多个步骤，例如 `msc+sg+d1`。其中：

- `mms`：最大最小归一化。
- `ss`：标准化。
- `snv`：标准正态变量变换。
- `sg`：Savitzky-Golay 平滑。
- `msc`：多元散射校正。
- `d1` / `d2`：一阶 / 二阶差分。
- `wave`：小波去噪，需要安装 PyWavelets。

### 光谱图像编码

```python
ENCODING_METHOD = "rp"
```

支持：

- `rp`：Recurrence Plot，递归图。
- `gasf`：Gramian Angular Summation Field。
- `gadf`：Gramian Angular Difference Field。
- `mtf`：Markov Transition Field。

编码结果以 `.npy` 文件缓存。缓存清单包含数据形状、哈希值和编码参数；若数据或参数变化，程序会自动重建缓存。

### 模型结构

```python
IMAGE_BACKBONE = "resnet50"
SPECTRAL_BACKBONE = "attn_cnn"
FUSION_METHOD = "acgf"
FEATURE_DIM = 512
```

图像分支支持：

```text
resnet50, mobilenet_v2, alexnet, vgg16, shufflenet_v2
```

光谱分支支持：

```text
cnn1d, attn_cnn, lstm, bilstm
```

融合方式支持：

```text
concat, dwgff, acgf
```

其中 `acgf` 是当前默认的轻量自适应通道门控融合模块，结合模态交互、模态差异和残差保留来生成融合特征。

### 最终分类器

```python
FINAL_CLASSIFIER = "linear_svm"
FINAL_CLASSIFIER_FEATURE_MODE = "hybrid"
```

支持：

```text
pso_svm, linear_svm, logreg, knn, rf, pls_da, fc
```

- `fc`：直接使用深度模型分类头输出。
- `linear_svm`：线性 SVM，当前默认。
- `pso_svm`：用粒子群优化搜索 SVM 的 C、gamma 和特征数。
- `logreg`、`knn`、`rf`、`pls_da`：传统机器学习分类器。

特征组合方式：

- `fused`：仅使用融合特征。
- `all`：拼接融合特征、图像特征、光谱特征。
- `hybrid`：在 `all` 基础上加入图像与光谱特征差值。
- `enhanced`：进一步加入乘积、均值差异等交互特征。

### 训练参数

```python
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-4
LR_SCHEDULER = "plateau"
EARLY_STOPPING_ENABLED = True
```

当前训练使用 AdamW、交叉熵损失、标签平滑、梯度裁剪、学习率调度和早停机制。可选 Center Loss 和分支辅助分类损失，其中当前默认启用图像分支与光谱分支辅助损失，未启用 Center Loss。

## 6. 运行方式

### 训练并评估默认实验

```bash
python main.py
```

### 运行方法组合烟雾测试

```bash
python validate_methods.py
```

烟雾测试会检查：

- 所有支持的图像编码方法能否生成二维矩阵。
- 图像骨干、光谱骨干、融合模块的组合能否前向传播。
- 若启用 Center Loss，则检查辅助损失是否为有效数值。
- 若干最终分类器能否完成基本训练与预测。

## 7. 实验套件开关

`mffn_config.py` 中提供了若干批量实验开关：

```python
RUN_PAPER_BENCHMARKS = False
RUN_TABLE2_ENCODING = False
RUN_TABLE3_IMAGE = False
RUN_TABLE4_SPECTRAL = False
RUN_TABLE5_FUSION_AND_CLASSIFIER = False
RUN_PREPROCESSING_BENCHMARKS = False
RUN_FEATURE_MODE_BENCHMARKS = False
```

用途：

- `RUN_TABLE2_ENCODING`：比较不同二维编码方法。
- `RUN_TABLE3_IMAGE`：比较不同图像骨干网络。
- `RUN_TABLE4_SPECTRAL`：比较不同光谱特征提取器。
- `RUN_TABLE5_FUSION_AND_CLASSIFIER`：比较融合方法与分类器。
- `RUN_PREPROCESSING_BENCHMARKS`：比较不同光谱预处理组合。
- `RUN_FEATURE_MODE_BENCHMARKS`：比较最终分类器特征组合方式。

这些开关适合做消融实验，但会显著增加训练时间。

## 8. 输出结果

训练完成后，主要输出位于 `results/`：

```text
results_summary.csv                         # 简要指标汇总
*_results_summary.csv                       # 单次实验详细指标
*_training_curves.png                       # 训练/验证损失与准确率曲线
*_confusion_matrix.png                      # 混淆矩阵
*_roc_curves.png                            # ROC 曲线
*_model.pth                                 # 最佳模型权重
*_classifier_comparison.csv                 # 分类器对比结果
paper_table*.csv                            # 论文表格式批量实验结果
```

当前已有一次实验结果，默认配置对应的测试集指标约为：

```text
Accuracy : 91.26%
Precision: 91.67%
Recall   : 91.29%
F1-score : 91.17%
```

## 9. 依赖环境

代码中直接使用的主要依赖包括：

```text
python
numpy
pandas
torch
torchvision
scikit-learn
scipy
matplotlib
Pillow
PyWavelets  # 仅 wave 预处理需要
```

如果 `IMAGE_PRETRAINED = True`，首次使用 torchvision 预训练模型时可能需要下载对应权重；在离线环境中可将其改为 `False`，或提前准备好权重缓存。

## 10. 注意事项

- `EVALUATION_PROTOCOL = "paper"` 会先过采样再划分数据，更贴近论文复现实验流程，但可能使合成样本信息进入测试分布；若用于真实泛化评估，建议使用 `strict`。
- `IMAGE_PRETRAINED = True` 时，图像骨干会加载 ImageNet 预训练权重；若网络不可用且本地无缓存，程序会报错。
- `NUM_WORKERS = 4` 在 Windows 上通常可用；若数据加载异常，可先改为 `0` 排查。
- `multimodal.py` 引用了当前目录中未提供的 `seq_encoder`、`attention`、`cnn` 模块，且不是当前主流程依赖文件，可视为历史代码或备用模型草稿。
- 批量基准实验会重复训练多个模型，建议确认 GPU、时间和磁盘空间后再开启。

## 11. 建议的实验记录方式

为了保证结果可追踪，建议每次实验至少记录：

- 使用的数据文件名与样本数。
- `SPECTRAL_PREPROCESSING_METHOD`。
- `ENCODING_METHOD`。
- `IMAGE_BACKBONE`、`SPECTRAL_BACKBONE`、`FUSION_METHOD`。
- `FINAL_CLASSIFIER` 与 `FINAL_CLASSIFIER_FEATURE_MODE`。
- `EVALUATION_PROTOCOL`。
- 输出的 `*_results_summary.csv`、混淆矩阵和 ROC 曲线。

项目已将多数关键配置写入实验名称中，便于从输出文件名直接追溯模型组合。
