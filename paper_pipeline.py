import csv
import gc
import hashlib
import json
import os
import glob
import random
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import mffn_config as cfg
from process import Preprocessing, SUPPORTED_PREPROCESSING_METHODS
from models.fusion import SUPPORTED_FUSION_METHODS, normalize_fusion_name
from models.image_backbones import SUPPORTED_IMAGE_BACKBONES, normalize_image_backbone_name
from models.losses import CenterLoss
from models.mffn_inirs import MFFNINIRS
from models.spectral_extractors import SUPPORTED_SPECTRAL_BACKBONES, normalize_spectral_backbone_name
from utils.image_encoding import (
    SUPPORTED_ENCODINGS,
    compute_rp_threshold,
    encode_series,
    normalize_encoding_method,
    resample_series,
)
from utils.oversampling import KMeansSMOTE
from utils.pso_svm import PSOSVM
from utils.wavelet_features import build_wavelet_views
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


RESULTS_DIR = cfg.RESULTS_DIR
ENCODING_ROOT = cfg.ENCODING_ROOT
SUPPORTED_FINAL_CLASSIFIERS = ("pso_svm", "linear_svm", "logreg", "knn", "rf", "pls_da", "fc")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENCODING_ROOT, exist_ok=True)


def normalize_preprocessing_method(method):
    if method is None:
        return "none"
    normalized = str(method).strip().lower()
    return normalized or "none"


def _preprocessing_steps(method):
    method = normalize_preprocessing_method(method)
    steps = [step.strip() for step in method.replace(",", "+").split("+") if step.strip()]
    return steps or ["none"]


def center_loss_enabled():
    return bool(getattr(cfg, "USE_CENTER_LOSS", False)) and float(getattr(cfg, "CENTER_LOSS_WEIGHT", 0.0)) > 0.0


def branch_aux_loss_enabled():
    image_weight = float(getattr(cfg, "IMAGE_AUX_LOSS_WEIGHT", 0.0))
    spectral_weight = float(getattr(cfg, "SPECTRAL_AUX_LOSS_WEIGHT", 0.0))
    return bool(getattr(cfg, "USE_BRANCH_AUX_LOSS", False)) and (image_weight > 0.0 or spectral_weight > 0.0)


def classifier_feature_selection_enabled():
    return bool(getattr(cfg, "CLASSIFIER_USE_FEATURE_SELECTION", False)) and int(
        getattr(cfg, "CLASSIFIER_MAX_FEATURES", 0)
    ) > 0


def _format_float_token(value):
    return f"{float(value):g}".replace(".", "p")


class SpectralImageDataset(Dataset):
    def __init__(self, spectral_data, labels, encoding_dir, split_prefix, encoding_method, image_size, load_images=True, wavelet_data=None):
        self.spectral_data = torch.as_tensor(spectral_data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.encoding_dir = encoding_dir
        self.split_prefix = split_prefix
        self.encoding_method = normalize_encoding_method(encoding_method)
        self.load_images = bool(load_images)
        self.image_size = int(image_size)
        self.wavelet_data = None if wavelet_data is None else torch.as_tensor(wavelet_data, dtype=torch.float32)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def _to_rgb_image(self, image_array):
        if self.encoding_method in {"gasf", "gadf"}:
            image_array = np.clip((image_array + 1.0) * 0.5, 0.0, 1.0)
        else:
            image_array = np.clip(image_array, 0.0, 1.0)
        image_uint8 = (image_array * 255.0).astype(np.uint8)
        image_rgb = np.stack([image_uint8, image_uint8, image_uint8], axis=2)
        return Image.fromarray(image_rgb)

    def __getitem__(self, idx):
        if self.wavelet_data is not None:
            image = self.wavelet_data[idx]
        elif self.load_images:
            image_path = os.path.join(self.encoding_dir, f"{self.split_prefix}_{idx}.npy")
            image_array = np.load(image_path).astype(np.float32)
            image = self.transform(self._to_rgb_image(image_array))
        else:
            image = torch.empty((3, self.image_size, self.image_size), dtype=torch.float32)
        spectral = self.spectral_data[idx].unsqueeze(0)
        label = self.labels[idx]
        return image, spectral, label


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((len(y), len(self.classes_)), dtype=np.float32)
        for idx, cls in enumerate(self.classes_):
            y_onehot[:, idx] = (y == cls).astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)
        n_components = min(self.n_components, X_scaled.shape[1], len(self.classes_) - 1 if len(self.classes_) > 1 else 1)
        self.model = PLSRegression(n_components=max(1, n_components))
        self.model.fit(X_scaled, y_onehot)
        return self

    def decision_function(self, X):
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float32))
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        prob = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return self.classes_[np.argmax(prob, axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config():
    modality_mode = str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower()
    if modality_mode not in ("multimodal", "wavelet_multiview", "spectral_only", "image_only"):
        raise ValueError("Unsupported MODALITY_MODE.")
    if not np.isclose(cfg.TRAIN_RATIO + cfg.VAL_RATIO + cfg.TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.")
    if cfg.ENCODING_METHOD not in SUPPORTED_ENCODINGS:
        raise ValueError(f"Unsupported ENCODING_METHOD: {cfg.ENCODING_METHOD}")
    if normalize_image_backbone_name(cfg.IMAGE_BACKBONE) not in SUPPORTED_IMAGE_BACKBONES:
        raise ValueError(f"Unsupported IMAGE_BACKBONE: {cfg.IMAGE_BACKBONE}")
    if normalize_spectral_backbone_name(cfg.SPECTRAL_BACKBONE) not in SUPPORTED_SPECTRAL_BACKBONES:
        raise ValueError(f"Unsupported SPECTRAL_BACKBONE: {cfg.SPECTRAL_BACKBONE}")
    if normalize_fusion_name(cfg.FUSION_METHOD) not in SUPPORTED_FUSION_METHODS:
        raise ValueError(f"Unsupported FUSION_METHOD: {cfg.FUSION_METHOD}")
    if cfg.FINAL_CLASSIFIER not in SUPPORTED_FINAL_CLASSIFIERS:
        raise ValueError(f"Unsupported FINAL_CLASSIFIER: {cfg.FINAL_CLASSIFIER}")
    if cfg.EVALUATION_PROTOCOL not in {"paper", "strict"}:
        raise ValueError("EVALUATION_PROTOCOL must be 'paper' or 'strict'.")
    if cfg.FINAL_CLASSIFIER_TRAIN_SPLIT not in {"auto", "train", "trainval"}:
        raise ValueError("FINAL_CLASSIFIER_TRAIN_SPLIT must be 'auto', 'train', or 'trainval'.")
    if str(getattr(cfg, "FINAL_CLASSIFIER_FEATURE_MODE", "fused")).lower() not in {"fused", "all", "hybrid", "enhanced"}:
        raise ValueError("FINAL_CLASSIFIER_FEATURE_MODE must be 'fused', 'all', 'hybrid', or 'enhanced'.")
    if cfg.LR_SCHEDULER.lower() not in {"step", "plateau", "cosine", "none"}:
        raise ValueError("LR_SCHEDULER must be 'step', 'plateau', 'cosine', or 'none'.")
    if cfg.BEST_MODEL_METRIC.lower() not in {"val_acc", "val_loss", "val_acc_or_loss"}:
        raise ValueError("BEST_MODEL_METRIC must be 'val_acc', 'val_loss', or 'val_acc_or_loss'.")
    if not 0.0 <= cfg.LABEL_SMOOTHING < 1.0:
        raise ValueError("LABEL_SMOOTHING must be in [0.0, 1.0).")
    if int(getattr(cfg, "FREEZE_IMAGE_BACKBONE_STAGES", 0)) < 0:
        raise ValueError("FREEZE_IMAGE_BACKBONE_STAGES must be >= 0.")
    if getattr(cfg, "IMAGE_DROPOUT", 0.0) < 0.0 or getattr(cfg, "IMAGE_DROPOUT", 0.0) >= 1.0:
        raise ValueError("IMAGE_DROPOUT must be in [0.0, 1.0).")
    if getattr(cfg, "FUSION_HIDDEN_DIM", 1) < 1:
        raise ValueError("FUSION_HIDDEN_DIM must be >= 1.")
    if not 0.0 < float(getattr(cfg, "FUSION_INITIAL_IMAGE_WEIGHT", 0.05)) < 1.0:
        raise ValueError("FUSION_INITIAL_IMAGE_WEIGHT must be between 0 and 1.")
    for name in ("SPECTRAL_BACKBONE_LR", "IMAGE_BACKBONE_LR", "FUSION_HEAD_LR"):
        if float(getattr(cfg, name, cfg.LEARNING_RATE)) <= 0.0:
            raise ValueError(f"{name} must be > 0.")
    if getattr(cfg, "FUSION_DROPOUT", 0.0) < 0.0 or getattr(cfg, "FUSION_DROPOUT", 0.0) >= 1.0:
        raise ValueError("FUSION_DROPOUT must be in [0.0, 1.0).")
    if getattr(cfg, "IMAGE_AUX_LOSS_WEIGHT", 0.0) < 0.0:
        raise ValueError("IMAGE_AUX_LOSS_WEIGHT must be >= 0.0.")
    if getattr(cfg, "SPECTRAL_AUX_LOSS_WEIGHT", 0.0) < 0.0:
        raise ValueError("SPECTRAL_AUX_LOSS_WEIGHT must be >= 0.0.")
    if int(getattr(cfg, "CLASSIFIER_MAX_FEATURES", 1)) < 1:
        raise ValueError("CLASSIFIER_MAX_FEATURES must be >= 1.")
    if getattr(cfg, "CENTER_LOSS_WEIGHT", 0.0) < 0.0:
        raise ValueError("CENTER_LOSS_WEIGHT must be >= 0.0.")
    if getattr(cfg, "CENTER_LOSS_LR", 0.0) <= 0.0:
        raise ValueError("CENTER_LOSS_LR must be > 0.0.")
    if int(getattr(cfg, "CENTER_LOSS_START_EPOCH", 1)) < 1:
        raise ValueError("CENTER_LOSS_START_EPOCH must be >= 1.")
    if center_loss_enabled() and int(getattr(cfg, "CENTER_LOSS_START_EPOCH", 1)) > int(cfg.EPOCHS):
        raise ValueError("CENTER_LOSS_START_EPOCH cannot exceed EPOCHS when center loss is enabled.")
    for step in _preprocessing_steps(cfg.SPECTRAL_PREPROCESSING_METHOD):
        if step not in SUPPORTED_PREPROCESSING_METHODS:
            raise ValueError(
                f"Unsupported SPECTRAL_PREPROCESSING_METHOD step: {step}. "
                f"Supported methods: {list(SUPPORTED_PREPROCESSING_METHODS)}"
            )
    for method in cfg.SPECTRAL_PREPROCESSING_METHODS:
        for step in _preprocessing_steps(method):
            if step not in SUPPORTED_PREPROCESSING_METHODS:
                raise ValueError(
                    f"Unsupported SPECTRAL_PREPROCESSING_METHODS step: {step}. "
                    f"Supported methods: {list(SUPPORTED_PREPROCESSING_METHODS)}"
                )
    if cfg.DATA_PATH and not os.path.exists(cfg.DATA_PATH):
        if not (os.path.exists(cfg.TRAIN_DATA_PATH) and os.path.exists(cfg.TEST_DATA_PATH)):
            raise FileNotFoundError("No valid dataset path found.")


def _is_float_token(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _csv_has_header(path):
    setting = cfg.CSV_HAS_HEADER
    if isinstance(setting, bool):
        return setting
    setting = str(setting).lower()
    if setting == "true":
        return True
    if setting == "false":
        return False
    if setting != "auto":
        raise ValueError("CSV_HAS_HEADER must be 'auto', True, or False.")
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row:
                return not all(_is_float_token(cell) for cell in row)
    raise ValueError(f"CSV file is empty: {path}")


def _read_csv(path):
    has_header = _csv_has_header(path)
    df = pd.read_csv(path, header=0 if has_header else None)
    y = pd.to_numeric(df.iloc[:, 0], errors="raise").to_numpy()
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    return X, y


def _encode_labels(labels):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    return encoded, unique_labels


def load_raw_dataset():
    if cfg.DATA_PATH and os.path.exists(cfg.DATA_PATH):
        X, y_raw = _read_csv(cfg.DATA_PATH)
        data_source = cfg.DATA_PATH
    else:
        X_train, y_train = _read_csv(cfg.TRAIN_DATA_PATH)
        X_test, y_test = _read_csv(cfg.TEST_DATA_PATH)
        X = np.vstack([X_train, X_test])
        y_raw = np.concatenate([y_train, y_test])
        data_source = f"{cfg.TRAIN_DATA_PATH} + {cfg.TEST_DATA_PATH}"

    y, class_labels = _encode_labels(y_raw)
    original_length = X.shape[1]
    if cfg.SPECTRAL_TARGET_LENGTH and original_length != cfg.SPECTRAL_TARGET_LENGTH:
        X = np.vstack([resample_series(row, cfg.SPECTRAL_TARGET_LENGTH) for row in X]).astype(np.float32)

    num_classes = len(np.unique(y))
    print("=" * 70)
    print("Paper reproduction dataset check")
    print("=" * 70)
    print(f"  Source: {data_source}")
    print(f"  Samples: {len(y)}")
    print(f"  Classes: {num_classes}")
    print(f"  Spectral length before alignment: {original_length}")
    print(f"  Spectral length used by pipeline: {X.shape[1]}")
    print(f"  Class labels: {class_labels.tolist()}")
    if num_classes != cfg.PAPER_EXPECTED_NUM_CLASSES or X.shape[1] != cfg.PAPER_EXPECTED_SPECTRAL_LENGTH:
        message = (
            "Current dataset does not match the paper dataset "
            f"(expected {cfg.PAPER_EXPECTED_NUM_CLASSES} classes and "
            f"{cfg.PAPER_EXPECTED_SPECTRAL_LENGTH} bands)."
        )
        if cfg.STRICT_PAPER_DATASET:
            raise ValueError(message)
        print(f"  Warning: {message}")
        print("  This run can reproduce the paper pipeline, but not the paper's reported accuracy.")

    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f"    Class {int(cls)}: {int(count)}")
    return X.astype(np.float32), y.astype(np.int64), class_labels


def apply_paper_oversampling(X, y):
    if not cfg.APPLY_OVERSAMPLING:
        return X, y

    print("\n" + "=" * 70)
    print("KMeans-SMOTE oversampling")
    print("=" * 70)
    sampler = KMeansSMOTE(
        k_neighbors=cfg.KMEANS_SMOTE_K_NEIGHBORS,
        k_clusters=cfg.KMEANS_SMOTE_K_CLUSTERS,
        random_state=cfg.RANDOM_SEED,
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    classes, counts = np.unique(y_resampled, return_counts=True)
    print(f"  Balanced samples: {len(y_resampled)}")
    for cls, count in zip(classes, counts):
        print(f"    Class {int(cls)}: {int(count)}")
    return X_resampled.astype(np.float32), y_resampled.astype(np.int64)


def apply_spectral_preprocessing_to_splits(splits):
    method = normalize_preprocessing_method(cfg.SPECTRAL_PREPROCESSING_METHOD)
    steps = _preprocessing_steps(method)
    if steps == ["none"]:
        return splits

    print("\n" + "=" * 70)
    print("Spectral preprocessing")
    print("=" * 70)
    print(f"  Method: {method}")

    processed = {
        split_name: (X_split.astype(np.float32, copy=True), y_split)
        for split_name, (X_split, y_split) in splits.items()
        if split_name in {"train", "val", "test"}
    }
    for step in steps:
        if step == "none":
            continue
        if step == "mms":
            scaler = MinMaxScaler().fit(processed["train"][0])
            for split_name, (X_split, y_split) in processed.items():
                processed[split_name] = (scaler.transform(X_split).astype(np.float32), y_split)
        elif step == "ss":
            scaler = StandardScaler().fit(processed["train"][0])
            for split_name, (X_split, y_split) in processed.items():
                processed[split_name] = (scaler.transform(X_split).astype(np.float32), y_split)
        elif step == "msc":
            reference = np.mean(processed["train"][0], axis=0).astype(np.float32)
            for split_name, (X_split, y_split) in processed.items():
                processed[split_name] = (_msc_with_reference(X_split, reference), y_split)
        else:
            for split_name, (X_split, y_split) in processed.items():
                processed[split_name] = (Preprocessing(step, X_split).astype(np.float32), y_split)

    processed["trainval"] = (
        np.vstack([processed["train"][0], processed["val"][0]]),
        np.concatenate([processed["train"][1], processed["val"][1]]),
    )
    for split_name in ("train", "val", "test"):
        original_bands = splits[split_name][0].shape[1]
        processed_bands = processed[split_name][0].shape[1]
        print(f"  {split_name}: {original_bands} -> {processed_bands} bands")
    return processed


def _msc_with_reference(X, reference):
    X = np.asarray(X, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    ref_centered = reference - reference.mean()
    ref_var = float(np.dot(ref_centered, ref_centered))
    if np.isclose(ref_var, 0.0):
        raise ValueError("MSC failed: training reference spectrum has zero variance.")

    corrected = np.empty_like(X, dtype=np.float32)
    ref_mean = float(reference.mean())
    for idx, spectrum in enumerate(X):
        spectrum_mean = float(spectrum.mean())
        slope = float(np.dot(ref_centered, spectrum - spectrum_mean) / ref_var)
        if np.isclose(slope, 0.0):
            raise ValueError(f"MSC failed for sample {idx}: regression slope is zero.")
        intercept = spectrum_mean - slope * ref_mean
        corrected[idx] = (spectrum - intercept) / slope
    return corrected.astype(np.float32)


def oversample_train_split(splits):
    if not cfg.APPLY_OVERSAMPLING:
        return splits
    X_train, y_train = splits["train"]
    X_train_resampled, y_train_resampled = apply_paper_oversampling(X_train, y_train)
    updated = dict(splits)
    updated["train"] = (X_train_resampled, y_train_resampled)
    updated["trainval"] = (
        np.vstack([X_train_resampled, updated["val"][0]]),
        np.concatenate([y_train_resampled, updated["val"][1]]),
    )
    return updated


def prepare_splits(X_raw, y_raw):
    protocol = cfg.EVALUATION_PROTOCOL
    print("\n" + "=" * 70)
    print(f"Evaluation protocol: {protocol}")
    print("=" * 70)
    if protocol == "paper":
        X_work, y_work = apply_paper_oversampling(X_raw, y_raw)
        splits = split_dataset(X_work, y_work)
        splits = apply_spectral_preprocessing_to_splits(splits)
        return splits
    if protocol == "strict":
        splits = split_dataset(X_raw, y_raw)
        splits = apply_spectral_preprocessing_to_splits(splits)
        splits = oversample_train_split(splits)
        return splits
    raise ValueError("EVALUATION_PROTOCOL must be 'paper' or 'strict'.")


def split_dataset(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_RATIO,
        random_state=cfg.RANDOM_SEED,
        stratify=y,
    )
    val_ratio_within_trainval = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio_within_trainval,
        random_state=cfg.RANDOM_SEED,
        stratify=y_trainval,
    )
    print("\n" + "=" * 70)
    print("Paper split (7:2:1)")
    print("=" * 70)
    print(f"  Train: {len(y_train)}")
    print(f"  Val:   {len(y_val)}")
    print(f"  Test:  {len(y_test)}")
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "trainval": (X_trainval, y_trainval),
    }


def get_encoding_dir(encoding_method):
    cache_name = normalize_encoding_method(encoding_method)
    if cfg.ENCODING_CACHE_TAG:
        cache_name = f"{cache_name}_{cfg.ENCODING_CACHE_TAG}"
    encoding_dir = os.path.join(ENCODING_ROOT, cache_name)
    os.makedirs(encoding_dir, exist_ok=True)
    return encoding_dir


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _fingerprint_array(array):
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.view(np.uint8))
    return hasher.hexdigest()


def _manifest_path(encoding_dir, prefix):
    return os.path.join(encoding_dir, f"{prefix}_manifest.json")


def _expected_manifest(X, prefix, encoding_method, encoding_params):
    return {
        "prefix": prefix,
        "method": normalize_encoding_method(encoding_method),
        "shape": [int(dim) for dim in X.shape],
        "sha256": _fingerprint_array(X),
        "params": _json_safe(encoding_params),
    }


def _clear_cached_prefix(encoding_dir, prefix):
    prefix_token = f"{prefix}_"
    for filename in os.listdir(encoding_dir):
        if filename.startswith(prefix_token) and filename.endswith(".npy"):
            os.remove(os.path.join(encoding_dir, filename))
    manifest_path = _manifest_path(encoding_dir, prefix)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)


def _cache_matches(encoding_dir, prefix, manifest):
    manifest_path = _manifest_path(encoding_dir, prefix)
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path, "r", encoding="utf-8") as handle:
        cached = json.load(handle)
    if cached != manifest:
        return False
    for idx in range(manifest["shape"][0]):
        if not os.path.exists(os.path.join(encoding_dir, f"{prefix}_{idx}.npy")):
            return False
    return True


def build_encoding_params(encoding_method, X_reference):
    encoding_method = normalize_encoding_method(encoding_method)
    params = {"target_length": cfg.ENCODING_TARGET_LENGTH}
    if encoding_method == "rp":
        params.update({
            "rp_threshold": compute_rp_threshold(
                X_reference,
                percentile=cfg.RP_THRESHOLD_PERCENTILE,
                m=cfg.RP_M,
                tau=cfg.RP_TAU,
                random_state=cfg.RANDOM_SEED,
                target_length=cfg.ENCODING_TARGET_LENGTH,
            ),
            "rp_m": cfg.RP_M,
            "rp_tau": cfg.RP_TAU,
        })
    elif encoding_method == "mtf":
        params["mtf_bins"] = cfg.MTF_BINS
    return params


def build_encoding_cache_prefix(encoding_method, split_name):
    parts = [normalize_encoding_method(encoding_method), cfg.EVALUATION_PROTOCOL, str(split_name).lower()]
    preprocessing_method = normalize_preprocessing_method(cfg.SPECTRAL_PREPROCESSING_METHOD)
    if preprocessing_method != "none":
        parts.append(preprocessing_method.replace("+", "_"))
    if cfg.ENCODING_CACHE_TAG:
        parts.append(str(cfg.ENCODING_CACHE_TAG))
    return "_".join(parts)


def generate_encoded_images(X, prefix, encoding_method, encoding_dir, encoding_params):
    manifest = _expected_manifest(X, prefix, encoding_method, encoding_params)
    if _cache_matches(encoding_dir, prefix, manifest):
        print(f"  Cache valid for {prefix}: {len(X)} images")
        return
    if cfg.REGENERATE_ENCODINGS_ON_MISMATCH:
        _clear_cached_prefix(encoding_dir, prefix)
    else:
        raise RuntimeError(f"Encoding cache mismatch for prefix '{prefix}'.")

    print(f"  Generating {encoding_method.upper()} images for {prefix}: {len(X)}")
    for idx, row in enumerate(X):
        encoded = encode_series(row, method=encoding_method, **encoding_params)
        np.save(os.path.join(encoding_dir, f"{prefix}_{idx}.npy"), encoded)
        if (idx + 1) % 100 == 0:
            print(f"    Saved {idx + 1}/{len(X)}")
            gc.collect()

    with open(_manifest_path(encoding_dir, prefix), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def build_dataloaders(splits, encoding_method, experiment_name):
    encoding_dir = get_encoding_dir(encoding_method)
    modality_mode = str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower()
    use_wavelets = modality_mode == "wavelet_multiview"
    load_images = modality_mode not in ("spectral_only", "wavelet_multiview")
    if cfg.EVALUATION_PROTOCOL == "paper":
        X_reference = np.vstack([splits["train"][0], splits["val"][0], splits["test"][0]])
    else:
        X_reference = splits["train"][0]
    if load_images:
        encoding_params = build_encoding_params(encoding_method, X_reference)
        print("\n" + "=" * 70)
        print("Image encoding")
        print("=" * 70)
        for split_name in ("train", "val", "test"):
            prefix = build_encoding_cache_prefix(encoding_method, split_name)
            X_split, _ = splits[split_name]
            generate_encoded_images(X_split, prefix, encoding_method, encoding_dir, encoding_params)
    else:
        print("\nImage encoding skipped: spectral/wavelet mode")

    dataloaders = {}
    for split_name in ("train", "val", "test"):
        prefix = build_encoding_cache_prefix(encoding_method, split_name)
        X_split, y_split = splits[split_name]
        wavelet_data = None
        if use_wavelets:
            wavelet_data = build_wavelet_views(
                X_split,
                wavelet=getattr(cfg, "WAVELET_NAME", "db4"),
                level=getattr(cfg, "WAVELET_LEVEL", 3),
                include_denoised=getattr(cfg, "WAVELET_INCLUDE_DENOISED", False),
            )
        dataset = SpectralImageDataset(
            X_split,
            y_split,
            encoding_dir=encoding_dir,
            split_prefix=prefix,
            encoding_method=encoding_method,
            image_size=cfg.IMAGE_SIZE,
            load_images=load_images,
            wavelet_data=wavelet_data,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=(split_name == "train"),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )
    return dataloaders


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def forward_model_outputs(model, images, spectra):
    outputs = model(images, spectra, return_dict=True)
    if not isinstance(outputs, dict):
        raise RuntimeError("Model must return a dict when return_dict=True.")
    required_keys = ("logits", "features", "image_features", "spectral_features")
    missing = [key for key in required_keys if key not in outputs]
    if missing:
        raise RuntimeError(
            "Model outputs missing required keys when return_dict=True: "
            f"{missing}"
        )
    if branch_aux_loss_enabled():
        aux_keys = ("image_logits", "spectral_logits")
        missing_aux = [key for key in aux_keys if key not in outputs]
        if missing_aux:
            raise RuntimeError(f"Branch auxiliary logits missing from model outputs: {missing_aux}")
    return outputs


def use_center_loss_for_epoch(epoch_number):
    return center_loss_enabled() and int(epoch_number) >= int(getattr(cfg, "CENTER_LOSS_START_EPOCH", 1))


def compute_branch_auxiliary_loss(outputs, labels, criterion):
    logits = outputs["logits"]
    branch_loss = logits.new_zeros(())
    if not branch_aux_loss_enabled():
        return branch_loss

    image_weight = float(getattr(cfg, "IMAGE_AUX_LOSS_WEIGHT", 0.0))
    spectral_weight = float(getattr(cfg, "SPECTRAL_AUX_LOSS_WEIGHT", 0.0))
    if image_weight > 0.0 and outputs["image_logits"] is not None:
        branch_loss = branch_loss + image_weight * criterion(outputs["image_logits"], labels)
    if spectral_weight > 0.0 and outputs["spectral_logits"] is not None:
        branch_loss = branch_loss + spectral_weight * criterion(outputs["spectral_logits"], labels)
    return branch_loss


def compute_objective(outputs, labels, criterion, auxiliary_criterion=None, auxiliary_weight=0.0, use_auxiliary=False):
    logits = outputs["logits"]
    cls_loss = criterion(logits, labels)
    metric_aux_loss = logits.new_zeros(())
    if auxiliary_criterion is not None and use_auxiliary:
        metric_aux_loss = auxiliary_criterion(outputs["features"], labels)
    branch_aux_loss = compute_branch_auxiliary_loss(outputs, labels, criterion)
    aux_loss = float(auxiliary_weight) * metric_aux_loss + branch_aux_loss
    total_loss = cls_loss + aux_loss
    return logits, total_loss, cls_loss, aux_loss


def evaluate_model(model, dataloader, criterion, device, auxiliary_criterion=None, auxiliary_weight=0.0, use_auxiliary=False):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_aux_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, spectra, labels in dataloader:
            images = images.to(device)
            spectra = spectra.to(device)
            labels = labels.to(device)
            outputs = forward_model_outputs(model, images, spectra)
            logits, loss, cls_loss, aux_loss = compute_objective(
                outputs,
                labels,
                criterion,
                auxiliary_criterion=auxiliary_criterion,
                auxiliary_weight=auxiliary_weight,
                use_auxiliary=use_auxiliary,
            )
            preds = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_aux_loss += aux_loss.item() * batch_size
            total_samples += batch_size
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    metrics = compute_metrics(np.asarray(all_labels), np.asarray(all_preds))
    metrics["Loss"] = total_loss / total_samples if total_samples else 0.0
    metrics["ClsLoss"] = total_cls_loss / total_samples if total_samples else 0.0
    metrics["AuxLoss"] = total_aux_loss / total_samples if total_samples else 0.0
    return metrics


def build_training_scheduler(optimizer):
    scheduler_name = cfg.LR_SCHEDULER.lower()
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.STEP_LR_STEP_SIZE,
            gamma=cfg.STEP_LR_GAMMA,
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.PLATEAU_LR_FACTOR,
            patience=cfg.PLATEAU_LR_PATIENCE,
            min_lr=cfg.MIN_LR,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.EPOCHS,
            eta_min=cfg.MIN_LR,
        )
    return None


def step_training_scheduler(scheduler, val_metrics):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_metrics["Loss"])
    else:
        scheduler.step()


def should_update_best_model(val_acc, val_loss, best_val_acc, best_val_loss):
    min_delta = cfg.EARLY_STOPPING_MIN_DELTA
    acc_improved = val_acc > best_val_acc + min_delta
    loss_improved = val_loss < best_val_loss - min_delta
    metric = cfg.BEST_MODEL_METRIC.lower()

    if metric == "val_acc":
        return acc_improved, acc_improved, loss_improved
    if metric == "val_loss":
        return loss_improved, acc_improved, loss_improved

    loss_acceptably_better = loss_improved and val_acc >= best_val_acc - cfg.VAL_LOSS_ACC_TOLERANCE
    return acc_improved or loss_acceptably_better, acc_improved, loss_improved


def build_auxiliary_objective(model, device):
    if not center_loss_enabled():
        return None, None, "none"
    objective = CenterLoss(
        num_classes=model.classifier.out_features,
        feat_dim=model.fused_dim,
        normalize=getattr(cfg, "CENTER_LOSS_NORMALIZE", True),
    ).to(device)
    optimizer = torch.optim.Adam(objective.parameters(), lr=cfg.CENTER_LOSS_LR)
    return objective, optimizer, "center"


def resolve_spectral_pretrained_path():
    configured = str(getattr(cfg, "SPECTRAL_PRETRAINED_PATH", "")).strip()
    if configured:
        path = configured if os.path.isabs(configured) else os.path.join(cfg.BASE_DIR, configured)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SPECTRAL_PRETRAINED_PATH does not exist: {path}")
        return path

    backbone = normalize_spectral_backbone_name(cfg.SPECTRAL_BACKBONE)
    preprocessing = normalize_preprocessing_method(cfg.SPECTRAL_PREPROCESSING_METHOD).replace("+", "_")
    pattern = os.path.join(RESULTS_DIR, f"*_{backbone}_*_spectral_only_*_{preprocessing}_*_model.pth")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No matching spectral-only checkpoint found. Set SPECTRAL_PRETRAINED_PATH to the "
            f"94.17% spectral model. Searched: {pattern}"
        )
    return candidates[0]


def load_spectral_pretrained_weights(model, device):
    if not bool(getattr(cfg, "LOAD_SPECTRAL_PRETRAINED", False)):
        return "", 0
    if model.spectral_branch is None:
        raise RuntimeError("LOAD_SPECTRAL_PRETRAINED requires an enabled spectral branch.")

    checkpoint_path = resolve_spectral_pretrained_path()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    prefix = "spectral_branch."
    spectral_state = {
        key[len(prefix):]: value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    if not spectral_state:
        raise RuntimeError(f"Checkpoint contains no '{prefix}*' parameters: {checkpoint_path}")
    incompatible = model.spectral_branch.load_state_dict(spectral_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Spectral checkpoint mismatch. Missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(f"  Loaded spectral pretrained weights: {checkpoint_path}")
    print(f"  Loaded spectral tensors: {len(spectral_state)}")
    return checkpoint_path, len(spectral_state)


def build_model_optimizer(model):
    groups = []
    assigned = set()

    def add_group(name, module, learning_rate):
        if module is None:
            return
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        parameters = [parameter for parameter in parameters if id(parameter) not in assigned]
        if not parameters:
            return
        assigned.update(id(parameter) for parameter in parameters)
        groups.append({"params": parameters, "lr": float(learning_rate), "group_name": name})

    secondary_lr = getattr(cfg, "WAVELET_BACKBONE_LR", cfg.LEARNING_RATE) \
        if str(getattr(cfg, "MODALITY_MODE", "")).lower() == "wavelet_multiview" \
        else getattr(cfg, "IMAGE_BACKBONE_LR", cfg.LEARNING_RATE)
    add_group("wavelet" if str(getattr(cfg, "MODALITY_MODE", "")).lower() == "wavelet_multiview" else "image", model.image_branch, secondary_lr)
    add_group("spectral", model.spectral_branch, getattr(cfg, "SPECTRAL_BACKBONE_LR", cfg.LEARNING_RATE))
    remaining = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    if remaining:
        groups.append({
            "params": remaining,
            "lr": float(getattr(cfg, "FUSION_HEAD_LR", cfg.LEARNING_RATE)),
            "group_name": "fusion_head",
        })
    if not groups:
        raise RuntimeError("Model has no trainable parameters.")
    return torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)


def train_model(model, dataloaders, device, experiment_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = build_model_optimizer(model)
    scheduler = build_training_scheduler(optimizer)
    auxiliary_criterion, auxiliary_optimizer, auxiliary_name = build_auxiliary_objective(model, device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cls_loss": [],
        "val_cls_loss": [],
        "train_aux_loss": [],
        "val_aux_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_model_val_acc = -1.0
    best_model_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("Training paper-style MFFN-INIRS")
    print("=" * 70)
    print(
        f"  Scheduler: {cfg.LR_SCHEDULER} | "
        f"Best metric: {cfg.BEST_MODEL_METRIC} | "
        f"Early stopping: {cfg.EARLY_STOPPING_ENABLED}"
    )
    print("  Optimizer groups: " + ", ".join(
        f"{group.get('group_name', 'group')}={group['lr']:.2e}" for group in optimizer.param_groups
    ))
    if auxiliary_name != "none":
        print(
            f"  Auxiliary loss: {auxiliary_name} | "
            f"Weight: {cfg.CENTER_LOSS_WEIGHT} | "
            f"Start epoch: {cfg.CENTER_LOSS_START_EPOCH} | "
            f"Normalize: {cfg.CENTER_LOSS_NORMALIZE}"
        )
    if branch_aux_loss_enabled():
        print(
            f"  Branch auxiliary loss | Image weight: {float(getattr(cfg, 'IMAGE_AUX_LOSS_WEIGHT', 0.0)):.2f} | "
            f"Spectral weight: {float(getattr(cfg, 'SPECTRAL_AUX_LOSS_WEIGHT', 0.0)):.2f}"
        )
    for epoch in range(cfg.EPOCHS):
        model.train()
        model.keep_frozen_modules_in_eval()
        use_auxiliary = use_center_loss_for_epoch(epoch + 1)
        running_loss = 0.0
        running_cls_loss = 0.0
        running_aux_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, spectra, labels in dataloaders["train"]:
            images = images.to(device)
            spectra = spectra.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if auxiliary_optimizer is not None:
                auxiliary_optimizer.zero_grad()

            outputs = forward_model_outputs(model, images, spectra)
            logits, loss, cls_loss, aux_loss = compute_objective(
                outputs,
                labels,
                criterion,
                auxiliary_criterion=auxiliary_criterion,
                auxiliary_weight=cfg.CENTER_LOSS_WEIGHT,
                use_auxiliary=use_auxiliary,
            )
            loss.backward()
            if cfg.MAX_GRAD_NORM and cfg.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()
            if auxiliary_optimizer is not None and use_auxiliary:
                auxiliary_optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_cls_loss += cls_loss.item() * batch_size
            running_aux_loss += aux_loss.item() * batch_size
            running_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            running_samples += batch_size

        train_loss = running_loss / running_samples
        train_cls_loss = running_cls_loss / running_samples
        train_aux_loss = running_aux_loss / running_samples
        train_acc = running_correct / running_samples
        val_metrics = evaluate_model(
            model,
            dataloaders["val"],
            criterion,
            device,
            auxiliary_criterion=auxiliary_criterion,
            auxiliary_weight=cfg.CENTER_LOSS_WEIGHT,
            use_auxiliary=use_auxiliary,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["Loss"])
        history["train_cls_loss"].append(train_cls_loss)
        history["val_cls_loss"].append(val_metrics["ClsLoss"])
        history["train_aux_loss"].append(train_aux_loss)
        history["val_aux_loss"].append(val_metrics["AuxLoss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["Accuracy"])

        improved, acc_improved, loss_improved = should_update_best_model(
            val_metrics["Accuracy"],
            val_metrics["Loss"],
            best_val_acc,
            best_val_loss,
        )
        if acc_improved:
            best_val_acc = val_metrics["Accuracy"]
        if loss_improved:
            best_val_loss = val_metrics["Loss"]

        if improved:
            best_model_val_acc = val_metrics["Accuracy"]
            best_model_val_loss = val_metrics["Loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        elif epoch + 1 >= cfg.MIN_EPOCHS:
            epochs_without_improvement += 1

        current_lr = max(group["lr"] for group in optimizer.param_groups)
        print(
            f"  Epoch {epoch + 1:02d}/{cfg.EPOCHS} | "
            f"LR {current_lr:.2e} | "
            f"Train Loss {train_loss:.4f} (CE {train_cls_loss:.4f}, Aux {train_aux_loss:.4f}) | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_metrics['Loss']:.4f} (CE {val_metrics['ClsLoss']:.4f}, Aux {val_metrics['AuxLoss']:.4f}) | "
            f"Val Acc {val_metrics['Accuracy']:.4f} | "
            f"NoImprove {epochs_without_improvement}/{cfg.EARLY_STOPPING_PATIENCE}"
        )
        step_training_scheduler(scheduler, val_metrics)

        if (
            cfg.EARLY_STOPPING_ENABLED
            and epoch + 1 >= cfg.MIN_EPOCHS
            and epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE
        ):
            print(
                f"  Early stopping at epoch {epoch + 1}: "
                f"no validation improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model_path = os.path.join(RESULTS_DIR, f"{experiment_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    plot_training_curves(history, experiment_name)
    print(
        f"  Best model at epoch {best_epoch}: "
        f"Val Acc {best_model_val_acc:.4f}, Val Loss {best_model_val_loss:.4f}"
    )
    print(f"  Model saved to {model_path}")
    return model, history, best_epoch


def combine_classifier_features(fused_features, image_features, spectral_features):
    modality_mode = str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower()
    if modality_mode == "image_only":
        return image_features.astype(np.float32)
    if modality_mode == "spectral_only":
        return spectral_features.astype(np.float32)
    mode = str(getattr(cfg, "FINAL_CLASSIFIER_FEATURE_MODE", "fused")).lower()
    if mode == "fused":
        return fused_features.astype(np.float32)
    if mode == "all":
        return np.hstack([fused_features, image_features, spectral_features]).astype(np.float32)
    if mode == "hybrid":
        feature_blocks = [fused_features, image_features, spectral_features]
        if image_features.shape[1] == spectral_features.shape[1]:
            feature_blocks.append(np.abs(image_features - spectral_features))
        return np.hstack(feature_blocks).astype(np.float32)
    if mode == "enhanced":
        feature_blocks = [fused_features, image_features, spectral_features]
        if image_features.shape[1] == spectral_features.shape[1]:
            modal_mean = 0.5 * (image_features + spectral_features)
            feature_blocks.extend([
                np.abs(image_features - spectral_features),
                image_features * spectral_features,
            ])
            if fused_features.shape[1] == modal_mean.shape[1]:
                feature_blocks.append(np.abs(fused_features - modal_mean))
        return np.hstack(feature_blocks).astype(np.float32)
    raise ValueError("FINAL_CLASSIFIER_FEATURE_MODE must be 'fused', 'all', 'hybrid', or 'enhanced'.")


def extract_features(model, dataloader, device):
    model.eval()
    all_fused_features = []
    all_image_features = []
    all_spectral_features = []
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for images, spectra, labels in dataloader:
            images = images.to(device)
            spectra = spectra.to(device)
            outputs = forward_model_outputs(model, images, spectra)
            all_fused_features.append(outputs["fused_features"].cpu().numpy())
            if outputs["image_features"] is not None:
                all_image_features.append(outputs["image_features"].cpu().numpy())
            if outputs["spectral_features"] is not None:
                all_spectral_features.append(outputs["spectral_features"].cpu().numpy())
            all_labels.append(labels.numpy())
            all_logits.append(outputs["logits"].cpu().numpy())
    fused_features = np.vstack(all_fused_features)
    image_features = np.vstack(all_image_features) if all_image_features else np.empty((len(fused_features), 0), dtype=np.float32)
    spectral_features = np.vstack(all_spectral_features) if all_spectral_features else np.empty((len(fused_features), 0), dtype=np.float32)
    classifier_features = combine_classifier_features(fused_features, image_features, spectral_features)
    return {
        "classifier_features": classifier_features,
        "fused_features": fused_features,
        "image_features": image_features,
        "spectral_features": spectral_features,
        "labels": np.concatenate(all_labels),
        "logits": np.vstack(all_logits),
    }


def _min_cv_splits(labels, desired_splits=5):
    counts = np.bincount(labels)
    positive = counts[counts > 0]
    return max(2, min(desired_splits, int(positive.min())))


def classifier_training_data(train_features, train_labels, val_features, val_labels):
    split = cfg.FINAL_CLASSIFIER_TRAIN_SPLIT
    if split == "auto":
        split = "train" if cfg.EVALUATION_PROTOCOL == "paper" else "trainval"
    if split == "train":
        return train_features, train_labels, split
    if split == "trainval":
        return (
            np.vstack([train_features, val_features]),
            np.concatenate([train_labels, val_labels]),
            split,
        )
    raise ValueError("FINAL_CLASSIFIER_TRAIN_SPLIT must be 'auto', 'train', or 'trainval'.")


def select_best_estimator(candidates, X, y):
    best_name = None
    best_estimator = None
    best_score = -np.inf
    cv = StratifiedKFold(n_splits=_min_cv_splits(y, 5), shuffle=True, random_state=cfg.RANDOM_SEED)
    scoring = getattr(cfg, "CLASSIFIER_CV_SCORING", "accuracy")
    for name, estimator in candidates:
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1)
        mean_score = float(scores.mean())
        print(f"  {name} CV {scoring}: {mean_score:.4f}")
        if mean_score > best_score:
            best_name = name
            best_estimator = clone(estimator)
            best_score = mean_score
    best_estimator.fit(X, y)
    return best_name, best_estimator, best_score


def classifier_feature_selection_k(input_dim):
    input_dim = int(input_dim)
    max_features = int(getattr(cfg, "CLASSIFIER_MAX_FEATURES", input_dim))
    return max(1, min(max_features, input_dim))


def maybe_build_selector_step(input_dim):
    if not classifier_feature_selection_enabled():
        return None
    k = classifier_feature_selection_k(input_dim)
    if k >= int(input_dim):
        return None
    return (
        "selector",
        SelectKBest(
            score_func=partial(mutual_info_classif, random_state=cfg.RANDOM_SEED),
            k=k,
        ),
    )


def build_scaled_classifier_pipeline(clf, input_dim):
    steps = [("scaler", StandardScaler())]
    selector_step = maybe_build_selector_step(input_dim)
    if selector_step is not None:
        steps.append(selector_step)
    steps.append(("clf", clf))
    return Pipeline(steps)


def build_classifier_candidates(input_dim):
    linear_svm_candidates = [
        (
            f"LinearSVM_C{c:g}",
            build_scaled_classifier_pipeline(
                LinearSVC(C=c, class_weight="balanced", random_state=cfg.RANDOM_SEED, dual=False, max_iter=5000),
                input_dim,
            ),
        )
        for c in cfg.LINEAR_SVM_C
    ]
    logreg_candidates = [
        (
            f"LogReg_C{c:g}",
            build_scaled_classifier_pipeline(
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    random_state=cfg.RANDOM_SEED,
                    max_iter=5000,
                    multi_class="auto",
                ),
                input_dim,
            ),
        )
        for c in cfg.LOGREG_C
    ]
    knn_candidates = [
        (
            f"KNN_k{n}",
            build_scaled_classifier_pipeline(KNeighborsClassifier(n_neighbors=n), input_dim),
        )
        for n in cfg.KNN_NEIGHBORS
    ]
    rf_candidates = [
        (f"RF_trees{n}", RandomForestClassifier(n_estimators=n, random_state=cfg.RANDOM_SEED))
        for n in cfg.RF_TREES
    ]
    pls_candidates = [
        (f"PLSDA_comp{n}", PLSDAClassifier(n_components=n))
        for n in cfg.PLS_COMPONENTS
    ]
    return {
        "linear_svm": linear_svm_candidates,
        "logreg": logreg_candidates,
        "knn": knn_candidates,
        "rf": rf_candidates,
        "pls_da": pls_candidates,
    }


def evaluate_final_classifier(classifier_name, trainval_features, trainval_labels, test_features, test_labels, test_logits=None):
    classifier_name = classifier_name.lower()
    if classifier_name == "fc":
        y_pred = np.argmax(test_logits, axis=1)
        y_score = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        return "FC", compute_metrics(test_labels, y_pred), y_pred, y_score

    if classifier_name == "pso_svm":
        estimator = PSOSVM(
            n_particles=cfg.PSO_PARTICLES,
            max_iter=cfg.PSO_MAX_ITER,
            cv_folds=cfg.PSO_CV_FOLDS,
            random_state=cfg.RANDOM_SEED,
            patience=cfg.PSO_PATIENCE,
            min_features=cfg.PSO_MIN_FEATURES,
            max_features_cap=cfg.PSO_MAX_FEATURES,
            scoring=cfg.PSO_SCORING,
            n_jobs=cfg.PSO_N_JOBS,
        )
        estimator.fit(trainval_features, trainval_labels)
        y_pred = estimator.predict(test_features)
        y_score = estimator.decision_function(test_features)
        return "PSO-SVM", compute_metrics(test_labels, y_pred), y_pred, y_score

    classifier_candidates = build_classifier_candidates(trainval_features.shape[1])
    if classifier_name not in classifier_candidates:
        raise ValueError(f"Unsupported final classifier: {classifier_name}")
    if classifier_feature_selection_enabled():
        selected_k = classifier_feature_selection_k(trainval_features.shape[1])
        if selected_k < trainval_features.shape[1]:
            print(f"  Classifier feature selection: top {selected_k}/{trainval_features.shape[1]} features")

    best_name, estimator, cv_score = select_best_estimator(
        classifier_candidates[classifier_name],
        trainval_features,
        trainval_labels,
    )
    y_pred = estimator.predict(test_features)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(test_features)
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(test_features)
    else:
        y_score = None
    display_name = best_name.replace("_", "-")
    scoring = getattr(cfg, "CLASSIFIER_CV_SCORING", "accuracy")
    print(f"  Selected {display_name} with CV {scoring} {cv_score:.4f}")
    return display_name, compute_metrics(test_labels, y_pred), y_pred, y_score


def plot_training_curves(history, experiment_name):
    has_aux_curve = any(abs(value) > 1e-12 for value in history.get("train_aux_loss", [])) or any(
        abs(value) > 1e-12 for value in history.get("val_aux_loss", [])
    )
    if has_aux_curve:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ax1, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1, ax2 = axes
        ax3 = None
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot([value * 100 for value in history["train_acc"]], label="Train Acc")
    ax2.plot([value * 100 for value in history["val_acc"]], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if ax3 is not None:
        ax3.plot(history["train_aux_loss"], label="Train Aux")
        ax3.plot(history["val_aux_loss"], label="Val Aux")
        ax3.set_title("Auxiliary Loss")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, experiment_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_score, num_classes, experiment_name):
    if y_score is None:
        return
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        y_score = y_score[:, None]
    if y_score.shape[1] != num_classes:
        return

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_idx in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, cls_idx], y_score[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {cls_idx} (AUC={roc_auc:.3f})")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def summarize_results(experiment_name, classifier_name, test_labels, y_pred, y_score):
    metrics = compute_metrics(test_labels, y_pred)
    cm = confusion_matrix(test_labels, y_pred)
    plot_confusion_matrix(cm, experiment_name)
    plot_roc_curves(test_labels, y_score, len(np.unique(test_labels)), experiment_name)
    results_df = pd.DataFrame({
        "Experiment": [experiment_name] * 4,
        "Classifier": [classifier_name] * 4,
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value (%)": [
            metrics["Accuracy"] * 100,
            metrics["Precision"] * 100,
            metrics["Recall"] * 100,
            metrics["F1-score"] * 100,
        ],
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, f"{experiment_name}_results_summary.csv"), index=False)
    print("\n" + "=" * 70)
    print("Test results")
    print("=" * 70)
    print(f"  Experiment: {experiment_name}")
    print(f"  Classifier: {classifier_name}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value * 100:.1f}%")
    print("  Confusion matrix:")
    print(cm)
    return metrics


def build_experiment_name(encoding_method, image_backbone, spectral_backbone, fusion_method, classifier_name=None):
    parts = [
        normalize_encoding_method(encoding_method),
        normalize_image_backbone_name(image_backbone),
        normalize_spectral_backbone_name(spectral_backbone),
        normalize_fusion_name(fusion_method),
        cfg.EVALUATION_PROTOCOL,
    ]
    modality_mode = str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower()
    if modality_mode == "wavelet_multiview":
        parts[0] = f"wavelet_{getattr(cfg, 'WAVELET_NAME', 'db4')}_l{int(getattr(cfg, 'WAVELET_LEVEL', 3))}"
        parts[1] = str(getattr(cfg, "WAVELET_BACKBONE", "attn_cnn")).lower()
    if modality_mode != "multimodal":
        parts.append(modality_mode)
    preprocessing_method = normalize_preprocessing_method(cfg.SPECTRAL_PREPROCESSING_METHOD)
    feature_dim = int(getattr(cfg, "FEATURE_DIM", 1024))
    if feature_dim != 1024:
        parts.append(f"feat{feature_dim}")
    if int(getattr(cfg, "FREEZE_IMAGE_BACKBONE_STAGES", 0)) > 0:
        parts.append(f"fz{int(getattr(cfg, 'FREEZE_IMAGE_BACKBONE_STAGES', 0))}")
    image_dropout = float(getattr(cfg, "IMAGE_DROPOUT", 0.0))
    if image_dropout > 0.0:
        parts.append(f"id{_format_float_token(image_dropout)}")
    fusion_dropout = float(getattr(cfg, "FUSION_DROPOUT", 0.0))
    if fusion_dropout > 0.0:
        parts.append(f"fd{_format_float_token(fusion_dropout)}")
    if normalize_fusion_name(fusion_method) in {"acgf", "spectral_residual"}:
        parts.append(f"h{int(getattr(cfg, 'FUSION_HIDDEN_DIM', 0))}")
    if normalize_fusion_name(fusion_method) == "spectral_residual":
        parts.append(f"iw{_format_float_token(getattr(cfg, 'FUSION_INITIAL_IMAGE_WEIGHT', 0.05))}")
    if bool(getattr(cfg, "LOAD_SPECTRAL_PRETRAINED", False)) and modality_mode != "image_only":
        parts.append("spft")
    if bool(getattr(cfg, "FREEZE_SPECTRAL_BACKBONE", False)) and modality_mode != "image_only":
        parts.append("spfreeze")
    if center_loss_enabled():
        center_token = f"cl{_format_float_token(cfg.CENTER_LOSS_WEIGHT)}"
        if getattr(cfg, "CENTER_LOSS_NORMALIZE", True):
            center_token += "n"
        parts.append(center_token)
        if int(getattr(cfg, "CENTER_LOSS_START_EPOCH", 1)) > 1:
            parts.append(f"warm{int(cfg.CENTER_LOSS_START_EPOCH)}")
    if branch_aux_loss_enabled():
        modality_mode = str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower()
        image_aux_weight = float(getattr(cfg, "IMAGE_AUX_LOSS_WEIGHT", 0.0))
        spectral_aux_weight = float(getattr(cfg, "SPECTRAL_AUX_LOSS_WEIGHT", 0.0))
        if image_aux_weight > 0.0 and modality_mode != "spectral_only":
            parts.append(f"iaux{_format_float_token(image_aux_weight)}")
        if spectral_aux_weight > 0.0 and modality_mode != "image_only":
            parts.append(f"saux{_format_float_token(spectral_aux_weight)}")
    feature_mode = str(getattr(cfg, "FINAL_CLASSIFIER_FEATURE_MODE", "fused")).lower()
    if feature_mode != "fused":
        parts.append(feature_mode)
    if classifier_feature_selection_enabled():
        parts.append(f"fs{int(getattr(cfg, 'CLASSIFIER_MAX_FEATURES', 0))}")
    if preprocessing_method != "none":
        parts.append(preprocessing_method.replace("+", "_"))
    if classifier_name:
        parts.append(classifier_name.lower())
    if cfg.EXPERIMENT_TAG:
        parts.append(cfg.EXPERIMENT_TAG)
    return "_".join(parts)


def train_and_evaluate_model(encoding_method, image_backbone, spectral_backbone, fusion_method, final_classifier):
    start_time = time.time()
    experiment_name = build_experiment_name(
        encoding_method,
        image_backbone,
        spectral_backbone,
        fusion_method,
        final_classifier,
    )

    X_raw, y_raw, class_labels = load_raw_dataset()
    splits = prepare_splits(X_raw, y_raw)
    dataloaders = build_dataloaders(splits, encoding_method, experiment_name)

    device = torch.device(cfg.DEVICE)
    model = MFFNINIRS(
        spectral_length=splits["train"][0].shape[1],
        num_classes=len(np.unique(y_raw)),
        feature_dim=cfg.FEATURE_DIM,
        image_backbone=image_backbone,
        spectral_backbone=spectral_backbone,
        fusion_method=fusion_method,
        image_pretrained=cfg.IMAGE_PRETRAINED,
        freeze_image_backbone_stages=int(getattr(cfg, "FREEZE_IMAGE_BACKBONE_STAGES", 0)),
        image_dropout=float(getattr(cfg, "IMAGE_DROPOUT", 0.0)),
        fusion_dropout=getattr(cfg, "FUSION_DROPOUT", 0.0),
        fusion_hidden_dim=getattr(cfg, "FUSION_HIDDEN_DIM", None),
        fusion_initial_image_weight=getattr(cfg, "FUSION_INITIAL_IMAGE_WEIGHT", 0.05),
        modality_mode=getattr(cfg, "MODALITY_MODE", "multimodal"),
        wavelet_backbone=getattr(cfg, "WAVELET_BACKBONE", "attn_cnn"),
        wavelet_in_channels=3 if bool(getattr(cfg, "WAVELET_INCLUDE_DENOISED", False)) else 2,
    ).to(device)

    spectral_checkpoint, loaded_spectral_tensors = load_spectral_pretrained_weights(model, device)
    if bool(getattr(cfg, "FREEZE_SPECTRAL_BACKBONE", False)):
        model.freeze_spectral_backbone()

    total_params = sum(param.numel() for param in model.parameters())
    print("\n" + "=" * 70)
    print("Model summary")
    print("=" * 70)
    print(f"  Experiment: {experiment_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Classes: {class_labels.tolist()}")
    print(f"  Spectral pretrained: {spectral_checkpoint or 'disabled'}")
    print(f"  Spectral backbone frozen: {bool(getattr(cfg, 'FREEZE_SPECTRAL_BACKBONE', False))}")
    print(
        f"  Secondary branch: {'wavelet' if str(getattr(cfg, 'MODALITY_MODE', '')).lower() == 'wavelet_multiview' else 'image'} | "
        f"Branch dropout: {float(getattr(cfg, 'IMAGE_DROPOUT', 0.0)):.2f} | "
        f"Fusion dropout: {float(getattr(cfg, 'FUSION_DROPOUT', 0.0)):.2f}"
    )

    model, history, best_epoch = train_model(model, dataloaders, device, experiment_name)
    final_image_weight = None
    if hasattr(model.fusion, "image_scale_logit"):
        final_image_weight = float(torch.sigmoid(model.fusion.image_scale_logit).detach().cpu())
        print(f"  Final residual image weight: {final_image_weight:.6f}")

    train_outputs = extract_features(model, dataloaders["train"], device)
    val_outputs = extract_features(model, dataloaders["val"], device)
    test_outputs = extract_features(model, dataloaders["test"], device)
    classifier_features, classifier_labels, classifier_train_split = classifier_training_data(
        train_outputs["classifier_features"],
        train_outputs["labels"],
        val_outputs["classifier_features"],
        val_outputs["labels"],
    )

    classifier_display_name, metrics, y_pred, y_score = evaluate_final_classifier(
        final_classifier,
        classifier_features,
        classifier_labels,
        test_outputs["classifier_features"],
        test_outputs["labels"],
        test_logits=test_outputs["logits"],
    )
    metrics = summarize_results(experiment_name, classifier_display_name, test_outputs["labels"], y_pred, y_score)
    elapsed_minutes = (time.time() - start_time) / 60.0
    return {
        "Experiment": experiment_name,
        "Encoding": normalize_encoding_method(encoding_method),
        "ImageBackbone": normalize_image_backbone_name(image_backbone),
        "SpectralBackbone": normalize_spectral_backbone_name(spectral_backbone),
        "Fusion": normalize_fusion_name(fusion_method),
        "ModalityMode": str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower(),
        "FeatureDim": int(getattr(cfg, "FEATURE_DIM", 0)),
        "FreezeImageBackboneStages": int(getattr(cfg, "FREEZE_IMAGE_BACKBONE_STAGES", 0)),
        "ImageDropout": float(getattr(cfg, "IMAGE_DROPOUT", 0.0)),
        "FusionDropout": float(getattr(cfg, "FUSION_DROPOUT", 0.0)),
        "SpectralPretrained": bool(getattr(cfg, "LOAD_SPECTRAL_PRETRAINED", False)),
        "SpectralCheckpoint": spectral_checkpoint,
        "LoadedSpectralTensors": loaded_spectral_tensors,
        "SpectralBackboneFrozen": bool(getattr(cfg, "FREEZE_SPECTRAL_BACKBONE", False)),
        "SpectralBackboneLR": float(getattr(cfg, "SPECTRAL_BACKBONE_LR", cfg.LEARNING_RATE)),
        "ImageBackboneLR": float(getattr(cfg, "IMAGE_BACKBONE_LR", cfg.LEARNING_RATE)),
        "FusionHeadLR": float(getattr(cfg, "FUSION_HEAD_LR", cfg.LEARNING_RATE)),
        "InitialImageWeight": float(getattr(cfg, "FUSION_INITIAL_IMAGE_WEIGHT", 0.0)),
        "FinalImageWeight": final_image_weight,
        "AuxLoss": "center" if center_loss_enabled() else "none",
        "CenterLossWeight": float(getattr(cfg, "CENTER_LOSS_WEIGHT", 0.0)) if center_loss_enabled() else 0.0,
        "CenterLossStartEpoch": int(getattr(cfg, "CENTER_LOSS_START_EPOCH", 1)) if center_loss_enabled() else 0,
        "CenterLossNormalize": bool(getattr(cfg, "CENTER_LOSS_NORMALIZE", False)),
        "BranchAuxLoss": "enabled" if branch_aux_loss_enabled() else "none",
        "ImageAuxLossWeight": float(getattr(cfg, "IMAGE_AUX_LOSS_WEIGHT", 0.0)) if branch_aux_loss_enabled() else 0.0,
        "SpectralAuxLossWeight": float(getattr(cfg, "SPECTRAL_AUX_LOSS_WEIGHT", 0.0)) if branch_aux_loss_enabled() else 0.0,
        "EvaluationProtocol": cfg.EVALUATION_PROTOCOL,
        "Preprocessing": normalize_preprocessing_method(cfg.SPECTRAL_PREPROCESSING_METHOD),
        "Classifier": classifier_display_name,
        "ClassifierFeatureMode": str(getattr(cfg, "FINAL_CLASSIFIER_FEATURE_MODE", "fused")).lower(),
        "ClassifierFeatureSelection": classifier_feature_selection_enabled(),
        "ClassifierMaxFeatures": int(getattr(cfg, "CLASSIFIER_MAX_FEATURES", 0)) if classifier_feature_selection_enabled() else 0,
        "ClassifierCVScoring": getattr(cfg, "CLASSIFIER_CV_SCORING", "accuracy"),
        "ClassifierTrainSplit": classifier_train_split,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1-score"],
        "BestEpoch": best_epoch,
        "TimeMinutes": elapsed_minutes,
    }


def train_and_compare_classifiers(encoding_method, image_backbone, spectral_backbone, fusion_method):
    X_raw, y_raw, _ = load_raw_dataset()
    splits = prepare_splits(X_raw, y_raw)
    experiment_stem = build_experiment_name(encoding_method, image_backbone, spectral_backbone, fusion_method, None)
    dataloaders = build_dataloaders(splits, encoding_method, experiment_stem)

    device = torch.device(cfg.DEVICE)
    model = MFFNINIRS(
        spectral_length=splits["train"][0].shape[1],
        num_classes=len(np.unique(y_raw)),
        feature_dim=cfg.FEATURE_DIM,
        image_backbone=image_backbone,
        spectral_backbone=spectral_backbone,
        fusion_method=fusion_method,
        image_pretrained=cfg.IMAGE_PRETRAINED,
        freeze_image_backbone_stages=int(getattr(cfg, "FREEZE_IMAGE_BACKBONE_STAGES", 0)),
        image_dropout=float(getattr(cfg, "IMAGE_DROPOUT", 0.0)),
        fusion_dropout=getattr(cfg, "FUSION_DROPOUT", 0.0),
        fusion_hidden_dim=getattr(cfg, "FUSION_HIDDEN_DIM", None),
    ).to(device)
    model, _, _ = train_model(model, dataloaders, device, experiment_stem)

    train_outputs = extract_features(model, dataloaders["train"], device)
    val_outputs = extract_features(model, dataloaders["val"], device)
    test_outputs = extract_features(model, dataloaders["test"], device)
    classifier_features, classifier_labels, _ = classifier_training_data(
        train_outputs["classifier_features"],
        train_outputs["labels"],
        val_outputs["classifier_features"],
        val_outputs["labels"],
    )

    rows = []
    for classifier_name in ("pso_svm", "linear_svm", "logreg", "knn", "rf", "pls_da", "fc"):
        display_name, metrics, y_pred, y_score = evaluate_final_classifier(
            classifier_name,
            classifier_features,
            classifier_labels,
            test_outputs["classifier_features"],
            test_outputs["labels"],
            test_logits=test_outputs["logits"],
        )
        experiment_name = build_experiment_name(
            encoding_method,
            image_backbone,
            spectral_backbone,
            fusion_method,
            classifier_name,
        )
        summarize_results(experiment_name, display_name, test_outputs["labels"], y_pred, y_score)
        rows.append({
            "Fusion method": normalize_fusion_name(fusion_method).upper(),
            "Classification": display_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-score": metrics["F1-score"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"{experiment_stem}_classifier_comparison.csv"), index=False)
    return df


def run_suite(experiments, summary_filename):
    results = []
    for experiment in experiments:
        result = train_and_evaluate_model(**experiment)
        results.append(result)
    summary = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, summary_filename), index=False)
    print(f"\nSummary saved to {os.path.join(RESULTS_DIR, summary_filename)}")
    return summary


def run_preprocessing_benchmarks():
    original_method = cfg.SPECTRAL_PREPROCESSING_METHOD
    results = []
    try:
        for method in cfg.SPECTRAL_PREPROCESSING_METHODS:
            cfg.SPECTRAL_PREPROCESSING_METHOD = normalize_preprocessing_method(method)
            print("\n" + "=" * 70)
            print(f"Preprocessing benchmark: {cfg.SPECTRAL_PREPROCESSING_METHOD}")
            print("=" * 70)
            result = train_and_evaluate_model(
                encoding_method=cfg.ENCODING_METHOD,
                image_backbone=cfg.IMAGE_BACKBONE,
                spectral_backbone=cfg.SPECTRAL_BACKBONE,
                fusion_method=cfg.FUSION_METHOD,
                final_classifier=cfg.FINAL_CLASSIFIER,
            )
            results.append(result)
    finally:
        cfg.SPECTRAL_PREPROCESSING_METHOD = original_method

    summary = pd.DataFrame(results).sort_values(by="F1", ascending=False)
    output_path = os.path.join(RESULTS_DIR, "preprocessing_benchmark.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nPreprocessing benchmark saved to {output_path}")
    print(summary[["Preprocessing", "Accuracy", "Precision", "Recall", "F1", "BestEpoch", "TimeMinutes"]])
    return summary


def run_paper_benchmarks():
    if cfg.RUN_TABLE2_ENCODING:
        experiments = [
            {
                "encoding_method": method,
                "image_backbone": "resnet50",
                "spectral_backbone": "cnn1d",
                "fusion_method": "dwgff",
                "final_classifier": "pso_svm",
            }
            for method in cfg.TABLE2_ENCODING_METHODS
        ]
        run_suite(experiments, "paper_table2_encoding.csv")

    if cfg.RUN_TABLE3_IMAGE:
        experiments = [
            {
                "encoding_method": "rp",
                "image_backbone": backbone,
                "spectral_backbone": "cnn1d",
                "fusion_method": "dwgff",
                "final_classifier": "pso_svm",
            }
            for backbone in cfg.TABLE3_IMAGE_BACKBONES
        ]
        run_suite(experiments, "paper_table3_image_backbone.csv")

    if cfg.RUN_TABLE4_SPECTRAL:
        experiments = [
            {
                "encoding_method": "rp",
                "image_backbone": "resnet50",
                "spectral_backbone": backbone,
                "fusion_method": "dwgff",
                "final_classifier": "pso_svm",
            }
            for backbone in cfg.TABLE4_SPECTRAL_BACKBONES
        ]
        run_suite(experiments, "paper_table4_spectral_backbone.csv")

    if cfg.RUN_TABLE5_FUSION_AND_CLASSIFIER:
        fusion_results = run_suite([
            {
                "encoding_method": "rp",
                "image_backbone": "resnet50",
                "spectral_backbone": "cnn1d",
                "fusion_method": "concat",
                "final_classifier": "pso_svm",
            },
            {
                "encoding_method": "rp",
                "image_backbone": "resnet50",
                "spectral_backbone": "cnn1d",
                "fusion_method": "dwgff",
                "final_classifier": "pso_svm",
            },
        ], "paper_table5_fusion.csv")
        classifier_results = train_and_compare_classifiers("rp", "resnet50", "cnn1d", "dwgff")
        classifier_results.to_csv(os.path.join(RESULTS_DIR, "paper_table5_classifier.csv"), index=False)
        print(fusion_results[["Fusion", "Classifier", "Accuracy", "Precision", "Recall", "F1"]])
        print(classifier_results)


def run_method_smoke_tests():
    dummy_series = np.linspace(0.0, 1.0, cfg.SPECTRAL_TARGET_LENGTH, dtype=np.float32)
    dummy_batch = np.stack([dummy_series, dummy_series[::-1]], axis=0)
    rp_params = build_encoding_params("rp", dummy_batch)
    for method in SUPPORTED_ENCODINGS:
        params = build_encoding_params(method, dummy_batch)
        encoded = encode_series(dummy_series, method=method, **params)
        if encoded.ndim != 2:
            raise RuntimeError(f"Encoding smoke test failed for {method}")

    image_input = torch.randn(2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    spectral_input = torch.randn(2, 1, cfg.SPECTRAL_TARGET_LENGTH)
    for image_backbone in SUPPORTED_IMAGE_BACKBONES:
        for spectral_backbone in SUPPORTED_SPECTRAL_BACKBONES:
            for fusion_method in SUPPORTED_FUSION_METHODS:
                model = MFFNINIRS(
                    spectral_length=cfg.SPECTRAL_TARGET_LENGTH,
                    num_classes=5,
                    feature_dim=cfg.FEATURE_DIM,
                    image_backbone=image_backbone,
                    spectral_backbone=spectral_backbone,
                    fusion_method=fusion_method,
                    image_pretrained=False,
                    freeze_image_backbone_stages=int(getattr(cfg, "FREEZE_IMAGE_BACKBONE_STAGES", 0)),
                    image_dropout=float(getattr(cfg, "IMAGE_DROPOUT", 0.0)),
                    fusion_dropout=getattr(cfg, "FUSION_DROPOUT", 0.0),
                    fusion_hidden_dim=getattr(cfg, "FUSION_HIDDEN_DIM", None),
                )
                with torch.no_grad():
                    logits = model(image_input, spectral_input)
                    features = model(image_input, spectral_input, return_features=True)
                    outputs = model(image_input, spectral_input, return_dict=True)
                if logits.shape[0] != 2 or features.shape[0] != 2:
                    raise RuntimeError("Model smoke test failed.")
                if outputs["logits"].shape != logits.shape or outputs["features"].shape != features.shape:
                    raise RuntimeError("Model return_dict smoke test failed.")
                if center_loss_enabled():
                    aux_criterion = CenterLoss(
                        num_classes=5,
                        feat_dim=model.fused_dim,
                        normalize=getattr(cfg, "CENTER_LOSS_NORMALIZE", True),
                    )
                    aux_loss = aux_criterion(outputs["features"], torch.tensor([0, 1], dtype=torch.long))
                    if not torch.isfinite(aux_loss):
                        raise RuntimeError("Center loss smoke test failed.")
                del model
                gc.collect()

    classifier_X = np.random.randn(20, 16).astype(np.float32)
    classifier_y = np.repeat(np.arange(5), 4)
    evaluate_final_classifier("knn", classifier_X, classifier_y, classifier_X, classifier_y)
    evaluate_final_classifier("rf", classifier_X, classifier_y, classifier_X, classifier_y)
    evaluate_final_classifier("pls_da", classifier_X, classifier_y, classifier_X, classifier_y)
    evaluate_final_classifier("pso_svm", classifier_X, classifier_y, classifier_X, classifier_y)
    print("Paper reproduction smoke tests passed.")



def run_feature_mode_benchmarks():
    original_mode = str(getattr(cfg, "FINAL_CLASSIFIER_FEATURE_MODE", "enhanced"))
    rows = []
    for mode in getattr(cfg, "FEATURE_MODE_CANDIDATES", ["all", "hybrid", "enhanced"]):
        cfg.FINAL_CLASSIFIER_FEATURE_MODE = mode
        result = train_and_evaluate_model(
            encoding_method=cfg.ENCODING_METHOD,
            image_backbone=cfg.IMAGE_BACKBONE,
            spectral_backbone=cfg.SPECTRAL_BACKBONE,
            fusion_method=cfg.FUSION_METHOD,
            final_classifier=cfg.FINAL_CLASSIFIER,
        )
        rows.append({
            "FeatureMode": mode,
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1": result["F1"],
            "BestEpoch": result["BestEpoch"],
            "Classifier": result["Classifier"],
        })
    cfg.FINAL_CLASSIFIER_FEATURE_MODE = original_mode
    summary = pd.DataFrame(rows)
    output_path = os.path.join(RESULTS_DIR, "feature_mode_benchmark.csv")
    summary.to_csv(output_path, index=False)
    print("\nFeature-mode benchmark saved to", output_path)
    print(summary)
def main():
    set_seed(cfg.RANDOM_SEED)
    validate_config()
    device = torch.device(cfg.DEVICE)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")

    if cfg.RUN_PREPROCESSING_BENCHMARKS:
        run_preprocessing_benchmarks()
        return

    if getattr(cfg, 'RUN_FEATURE_MODE_BENCHMARKS', False):
        run_feature_mode_benchmarks()
        return

    if cfg.RUN_PAPER_BENCHMARKS or cfg.RUN_TABLE2_ENCODING or cfg.RUN_TABLE3_IMAGE or cfg.RUN_TABLE4_SPECTRAL or cfg.RUN_TABLE5_FUSION_AND_CLASSIFIER:
        run_paper_benchmarks()
        return

    result = train_and_evaluate_model(
        encoding_method=cfg.ENCODING_METHOD,
        image_backbone=cfg.IMAGE_BACKBONE,
        spectral_backbone=cfg.SPECTRAL_BACKBONE,
        fusion_method=cfg.FUSION_METHOD,
        final_classifier=cfg.FINAL_CLASSIFIER,
    )
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value (%)": [
            result["Accuracy"] * 100,
            result["Precision"] * 100,
            result["Recall"] * 100,
            result["F1"] * 100,
        ],
    })
    modality_mode = str(getattr(cfg, "MODALITY_MODE", "multimodal")).lower()
    summary.to_csv(os.path.join(RESULTS_DIR, f"results_summary_{modality_mode}.csv"), index=False)


if __name__ == "__main__":
    main()






