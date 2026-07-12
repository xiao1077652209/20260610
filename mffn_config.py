import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODING_ROOT = os.path.join(BASE_DIR, "encodings")


#DATA_PATH = os.path.join(DATASET_DIR, "diesel_spec_with_label.csv")
DATA_PATH = os.path.join(DATASET_DIR, "20251203_535_产地_7.csv")
#TRAIN_DATA_PATH = os.path.join(DATASET_DIR, "20251203_535_产地_7_TRAIN.csv")
#TEST_DATA_PATH = os.path.join(DATASET_DIR, "20251203_535_产地_7_TEST.csv")
CSV_HAS_HEADER = "auto"  # auto / True / False

# Paper alignment
STRICT_PAPER_DATASET = False
PAPER_EXPECTED_NUM_CLASSES = 5
PAPER_EXPECTED_SPECTRAL_LENGTH = 778
SPECTRAL_TARGET_LENGTH = 778
ENCODING_TARGET_LENGTH = 778

#PAPER_EXPECTED_SPECTRAL_LENGTH = 401
#SPECTRAL_TARGET_LENGTH = 401
#ENCODING_TARGET_LENGTH = 401
SPECTRAL_PREPROCESSING_METHOD = "snv+d1+sg"  # follows the reference NIR branch order
SPECTRAL_PREPROCESSING_METHODS = [
    "msc+sg+d1",
    "snv+sg+d1",
]
EVALUATION_PROTOCOL = "paper"      # paper: oversample before split; strict: split before oversampling train only
EXPERIMENT_TAG = "paper_repro"
ENCODING_CACHE_TAG = "paper_rp401"

# Default model
MODALITY_MODE = "spectral_only"    # multimodal / spectral_only / image_only
ENCODING_METHOD = "rp"             # rp / gadf / gasf / mtf
IMAGE_BACKBONE = "resnet50"        # resnet50 / mobilenet_v2 / alexnet / vgg16 / shufflenet_v2
SPECTRAL_BACKBONE = "multiscale_cnn"  # multiscale_cnn / attn_cnn / cnn1d / lstm / bilstm
FUSION_METHOD = "acgf"             # dwgff / concat / acgf
FINAL_CLASSIFIER = "linear_svm"   # pso_svm / linear_svm / logreg / knn / rf / pls_da / fc
IMAGE_PRETRAINED = True
FREEZE_IMAGE_BACKBONE_STAGES = 2    # freeze slightly deeper visual stages to improve small-sample generalization
IMAGE_DROPOUT = 0.10                # slightly stronger regularization for small-sample generalization
APPLY_OVERSAMPLING = True
FINAL_CLASSIFIER_TRAIN_SPLIT = "trainval"  # auto / train / trainval
FINAL_CLASSIFIER_FEATURE_MODE = "hybrid"   # fused / all / hybrid / enhanced
FUSION_DROPOUT = 0.15
FUSION_HIDDEN_DIM = 256             # used by ACGF lightweight gated fusion
USE_CENTER_LOSS = False             # keep disabled on the mainline: current server result shows it hurts PSO-SVM test accuracy
CENTER_LOSS_WEIGHT = 0.03
CENTER_LOSS_LR = 1e-3
CENTER_LOSS_START_EPOCH = 4
CENTER_LOSS_NORMALIZE = True
USE_BRANCH_AUX_LOSS = True
USE_MODAL_ALIGN_LOSS = False
MODAL_ALIGN_LOSS_WEIGHT = 0.01
IMAGE_AUX_LOSS_WEIGHT = 0.05
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
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
STEP_LR_STEP_SIZE = 5
STEP_LR_GAMMA = 0.1
LR_SCHEDULER = "plateau"           # step / plateau / cosine / none
PLATEAU_LR_FACTOR = 0.5
PLATEAU_LR_PATIENCE = 8
MIN_LR = 1e-6
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 14
MIN_EPOCHS = 12
EARLY_STOPPING_MIN_DELTA = 1e-4
BEST_MODEL_METRIC = "val_acc_or_loss"  # val_acc / val_loss / val_acc_or_loss
VAL_LOSS_ACC_TOLERANCE = 0.01
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.05
FEATURE_DIM = 512
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
LINEAR_SVM_C = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
#LINEAR_SVM_C = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
LOGREG_C = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
CLASSIFIER_USE_FEATURE_SELECTION = True
CLASSIFIER_MAX_FEATURES = 512

KNN_NEIGHBORS = [3, 5, 7, 10, 12]
RF_TREES = [20, 50, 100]
PLS_COMPONENTS = [2, 4, 6, 8, 10]
