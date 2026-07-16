import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODING_ROOT = os.path.join(BASE_DIR, "encodings")


# ============================================================
# 数据集选择（只需改这一行）
# ============================================================
DATASET_NAME = "产地"  # 可选: 品种 / 产地 / Apple / Coffee / Mango / Diesel


# ============================================================
# 数据集注册表（每个数据集的专属配置）
# 模型支持变长输入，target_length 建议保留原始波段数
#
# 协议选择建议：
#   paper:  小样本 + 不均衡 → 先过采样再划分（与论文一致）
#   strict: 样本均衡或量大 → 先划分再仅对训练集过采样（更严谨）
#
# 大数据集建议：
#   large_data_mode=True 时，会自动跳过过采样、使用 strict 协议、
#   增大 batch_size、增加 epoch、使用更大特征维度
# ============================================================
DATASET_CONFIGS = {
    "品种": {
        "path": "20260109_286_品种_8.csv",
        "target_length": None,
        "oversampling": True,
        "strategy": "max",
        "protocol": "strict",
        # 大数据模式（默认关闭）
        "large_data_mode": False,
        "feature_dim": 256,
        "batch_size": 64,
        "epochs": 60,
    },
    "产地": {
        "path": "20251203_535_产地_7.csv",
        "target_length": None,
        "oversampling": True,
        "strategy": "max",              # median → max：增加训练样本量，缓解细粒度分类数据不足
        "protocol": "strict",
        "large_data_mode": False,
        "feature_dim": 256,
        "batch_size": 64,
        "epochs": 150,                  # 80 → 150：让模型充分训练
    },
    "Apple": {
        "path": "apple.csv",
        "target_length": 2151,
        "oversampling": False,
        "strategy": "max",
        "protocol": "strict",
        # Apple 数据集样本量较大且均衡，建议使用大数据模式
        "large_data_mode": True,
        "feature_dim": 1024,
        "batch_size": 128,
        "epochs": 100,
    },
    "Coffee": {
        "path": "coffee.csv",
        "target_length": 286,
        "oversampling": False,
        "strategy": "max",
        "protocol": "strict",
        # Coffee 数据集样本量较大且均衡，建议使用大数据模式
        "large_data_mode": True,
        "feature_dim": 1024,
        "batch_size": 128,
        "epochs": 100,
    },
    "Mango": {
        "path": "Mango.csv",
        "target_length": 281,
        "oversampling": True,
        "strategy": "median",
        "protocol": "strict",
        "large_data_mode": False,
        "feature_dim": 256,
        "batch_size": 64,
        "epochs": 60,
    },
    "Diesel": {
        "path": "diesel_spec_with_label.csv",
        "target_length": 401,
        "oversampling": True,
        "strategy": "max",
        "protocol": "strict",
        "large_data_mode": False,
        "feature_dim": 256,
        "batch_size": 64,
        "epochs": 60,
    },
}

# 自动应用所选数据集的配置
_selected = DATASET_CONFIGS[DATASET_NAME]
DATA_PATH = os.path.join(DATASET_DIR, _selected["path"])
SPECTRAL_TARGET_LENGTH = _selected["target_length"]
ENCODING_TARGET_LENGTH = _selected["target_length"]
APPLY_OVERSAMPLING = _selected["oversampling"]
OVERSAMPLING_STRATEGY = _selected["strategy"]
EVALUATION_PROTOCOL = _selected["protocol"]

# ---- 大数据模式配置 ----
# 当 large_data_mode=True 时，以下配置会被覆盖
LARGE_DATA_MODE = _selected.get("large_data_mode", False)
if LARGE_DATA_MODE:
    # 大数据模式下自动调整配置
    APPLY_OVERSAMPLING = False          # 大数据集通常不需要过采样
    EVALUATION_PROTOCOL = "strict"      # 大数据集使用 strict 协议更严谨
    FEATURE_DIM = _selected.get("feature_dim", 1024)
    BATCH_SIZE = _selected.get("batch_size", 128)
    EPOCHS = _selected.get("epochs", 100)
else:
    FEATURE_DIM = _selected.get("feature_dim", 512)
    BATCH_SIZE = _selected.get("batch_size", 64)
    EPOCHS = _selected.get("epochs", 60)


#TRAIN_DATA_PATH = os.path.join(DATASET_DIR, "20251203_535_产地_7_TRAIN.csv")
#TEST_DATA_PATH = os.path.join(DATASET_DIR, "20251203_535_产地_7_TEST.csv")
CSV_HAS_HEADER = "auto"  # auto / True / False

# Paper alignment
STRICT_PAPER_DATASET = False
PAPER_EXPECTED_NUM_CLASSES = None    # None: accept the class count of the selected dataset
PAPER_EXPECTED_SPECTRAL_LENGTH = 1555

#PAPER_EXPECTED_SPECTRAL_LENGTH = 401
SPECTRAL_PREPROCESSING_METHOD = "snv+d1+sg"  # follows the reference NIR branch order
SPECTRAL_PREPROCESSING_METHODS = [
    "msc+sg+d1",
    "snv+sg+d1",
]
EXPERIMENT_TAG = "paper_repro"
ENCODING_CACHE_TAG = "paper_rp1556"

# Default model
MODALITY_MODE = "wavelet_multiview"  # wavelet_multiview / multimodal / spectral_only / image_only
ENCODING_METHOD = "rp"             # rp / gadf / gasf / mtf
IMAGE_BACKBONE = "resnet50"        # resnet50 / mobilenet_v2 / alexnet / vgg16 / shufflenet_v2
SPECTRAL_BACKBONE = "multiscale_cnn"  # multiscale_cnn / attn_cnn / cnn1d / lstm / bilstm
FUSION_METHOD = "spectral_residual"  # spectral_residual / acgf / dwgff / concat
FINAL_CLASSIFIER = "linear_svm"   # pso_svm / linear_svm / logreg / knn / rf / pls_da / fc
IMAGE_PRETRAINED = False            # wavelet_multiview never uses image pretrained weights
FREEZE_IMAGE_BACKBONE_STAGES = 2    # freeze slightly deeper visual stages to improve small-sample generalization
IMAGE_DROPOUT = 0.10                # used only by the real image branch
FINAL_CLASSIFIER_TRAIN_SPLIT = "trainval"  # auto / train / trainval
FINAL_CLASSIFIER_FEATURE_MODE = "enhanced" # fused → enhanced：给分类器提供更丰富的特征组合
FUSION_DROPOUT = 0.20
FUSION_HIDDEN_DIM = 256             # used by ACGF lightweight gated fusion
FUSION_INITIAL_SECONDARY_WEIGHT = 0.10  # image or wavelet residual starts close to spectral-only
LOAD_SPECTRAL_PRETRAINED = False    # 先设为 False，等跑出光谱-only 模型后再改为 True
SPECTRAL_PRETRAINED_PATH = ""       # empty: auto-discover the matching spectral-only model in RESULTS_DIR
FREEZE_SPECTRAL_BACKBONE = False    # 先设为 False，等启用光谱预训练后再改为 True

SPECTRAL_BACKBONE_LR = 5e-5
IMAGE_BACKBONE_LR = 5e-5
FUSION_HEAD_LR = 5e-5
WAVELET_BACKBONE = "attn_cnn"
WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
WAVELET_INCLUDE_DENOISED = True      # True: A3 + mid-details + denoised reconstruction (3 channels)
WAVELET_BACKBONE_LR = 5e-5
WAVELET_AUX_LOSS_WEIGHT = 0.05
USE_CENTER_LOSS = True              # False → True：开启Center Loss，让同类特征更紧凑
CENTER_LOSS_WEIGHT = 0.03
CENTER_LOSS_LR = 1e-3
CENTER_LOSS_START_EPOCH = 5         # 从第5轮开始，让模型先学习基本分类
CENTER_LOSS_NORMALIZE = True
USE_BRANCH_AUX_LOSS = False         # True → False：关闭无效的分支aux loss，改用Center Loss
# USE_MODAL_ALIGN_LOSS = False       # 未实现，暂保留
# MODAL_ALIGN_LOSS_WEIGHT = 0.01     # 未实现，暂保留
IMAGE_AUX_LOSS_WEIGHT = 0.02         # used only by the real image branch
SPECTRAL_AUX_LOSS_WEIGHT = 0.10

# Reproduction suites
RUN_PAPER_BENCHMARKS = False
RUN_TABLE2_ENCODING = False
RUN_TABLE3_IMAGE = False
RUN_TABLE4_SPECTRAL = False
RUN_TABLE5_FUSION_AND_CLASSIFIER = False
RUN_PREPROCESSING_BENCHMARKS = False
RUN_FEATURE_MODE_BENCHMARKS = False
FEATURE_MODE_CANDIDATES = ["all", "hybrid", "enhanced"]

TABLE2_ENCODING_METHODS = ["gadf", "gasf", "mtf", "rp"]
TABLE3_IMAGE_BACKBONES = ["mobilenet_v2", "alexnet", "vgg16", "shufflenet_v2", "resnet50"]
TABLE4_SPECTRAL_BACKBONES = ["lstm", "bilstm", "cnn1d", "attn_cnn"]

# RP / MTF
RP_M = 1
RP_TAU = 1
RP_THRESHOLD_PERCENTILE = 10
MTF_BINS = 8
REGENERATE_ENCODINGS_ON_MISMATCH = True

# Oversampling
KMEANS_SMOTE_K_NEIGHBORS = 5
KMEANS_SMOTE_K_CLUSTERS = 8

# Training setup from paper
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True
# BATCH_SIZE 和 EPOCHS 已由大数据模式动态设置
# BATCH_SIZE = 64
# EPOCHS = 60
LEARNING_RATE = 1e-4                 # 5e-5 → 1e-4：提高学习率，加快收敛
WEIGHT_DECAY = 5e-4
STEP_LR_STEP_SIZE = 5
STEP_LR_GAMMA = 0.1
LR_SCHEDULER = "cosine"             # plateau → cosine：全程稳定降低学习率，更适合细粒度分类
PLATEAU_LR_FACTOR = 0.5
PLATEAU_LR_PATIENCE = 8
MIN_LR = 1e-6
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 25        # 10 → 25：增加耐心，避免验证集波动导致过早停止
MIN_EPOCHS = 30                     # 12 → 30：确保前期充分探索
EARLY_STOPPING_MIN_DELTA = 1e-4
BEST_MODEL_METRIC = "val_acc_or_loss"  # val_acc / val_loss / val_acc_or_loss
VAL_LOSS_ACC_TOLERANCE = 0.01
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.05
# FEATURE_DIM 已由大数据模式动态设置
# FEATURE_DIM = 512
IMAGE_SIZE = 224

# Final classifiers
PSO_PARTICLES = 8
PSO_MAX_ITER = 12
PSO_CV_FOLDS = 5
PSO_PATIENCE = 4
PSO_MIN_FEATURES = 32
PSO_MAX_FEATURES = 256
CLASSIFIER_CV_SCORING = "f1_macro"  # accuracy / balanced_accuracy / f1_macro
PSO_SCORING = CLASSIFIER_CV_SCORING
PSO_N_JOBS = -1
LINEAR_SVM_C = [0.1, 0.5, 1.0]
#LINEAR_SVM_C = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
LOGREG_C = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
CLASSIFIER_USE_FEATURE_SELECTION = True
CLASSIFIER_MAX_FEATURES = 256

KNN_NEIGHBORS = [3, 5, 7, 10, 12]
RF_TREES = [20, 50, 100]
PLS_COMPONENTS = [2, 4, 6, 8, 10]
