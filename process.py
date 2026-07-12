"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github :
    @WeChat : Fu_siry
    @License：

"""
import numpy as np
try:
    import pywt
except ImportError:
    pywt = None
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
#import pywt

# ref1: 湖南示范大学同学实列，并做了部分修改
# ref2: https://blog.csdn.net/qq2512446791

SUPPORTED_PREPROCESSING_METHODS = (
    "none",
    "mms",
    "ss",
    "ct",
    "snv",
    "ma",
    "sg",
    "msc",
    "d1",
    "d2",
    "dt",
    "wave",
)


def _as_2d_float_array(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D spectral data, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Spectral data contains NaN or inf.")
    return data.copy()


# 最大最小值归一化
def MMS(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MinMaxScaler :(n_samples, n_features)
       """
    data = _as_2d_float_array(data)
    return MinMaxScaler().fit_transform(data).astype(np.float32)


# 标准化
def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
       """
    data = _as_2d_float_array(data)
    return StandardScaler().fit_transform(data).astype(np.float32)


# 均值中心化
def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
       """
    data = _as_2d_float_array(data)
    return (data - np.mean(data, axis=1, keepdims=True)).astype(np.float32)


# 标准正态变换
def SNV(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after SNV :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    data_std = np.std(data, axis=1, keepdims=True)
    if np.any(np.isclose(data_std, 0.0)):
        raise ValueError("SNV cannot process spectra with zero standard deviation.")
    data_average = np.mean(data, axis=1, keepdims=True)
    return ((data - data_average) / data_std).astype(np.float32)



# 移动平均平滑
def MA(data, WSZ=11):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param WSZ: int
       :return: data after MA :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    if WSZ < 3 or WSZ % 2 == 0:
        raise ValueError(f"MA window size must be an odd integer >= 3, got {WSZ}")
    if WSZ > data.shape[1]:
        raise ValueError(f"MA window size {WSZ} exceeds spectral length {data.shape[1]}")
    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data.astype(np.float32)


# Savitzky-Golay平滑滤波
def SG(data, w=11, p=2):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    if w < 3 or w % 2 == 0:
        raise ValueError(f"SG window size must be an odd integer >= 3, got {w}")
    if p >= w:
        raise ValueError(f"SG polynomial order p must be smaller than window size w, got p={p}, w={w}")
    if w > data.shape[1]:
        raise ValueError(f"SG window size {w} exceeds spectral length {data.shape[1]}")
    return signal.savgol_filter(data, w, p, axis=1).astype(np.float32)


# 一阶导数
def D1(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after First derivative :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    if data.shape[1] < 2:
        raise ValueError("D1 requires at least 2 spectral features.")
    return np.diff(data, axis=1).astype(np.float32)


# 二阶导数
def D2(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after second derivative :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    if data.shape[1] < 3:
        raise ValueError("D2 requires at least 3 spectral features.")
    return np.diff(data, n=2, axis=1).astype(np.float32)


# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data, dtype=np.float32)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = float(l.coef_.ravel()[0])
        b = float(l.intercept_.ravel()[0])
        out[i] = out[i] - (x * k + b)

    return out.astype(np.float32)


# 多元散射校正
def MSC(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    data = _as_2d_float_array(data)
    n, p = data.shape
    msc = np.ones((n, p))

    mean = np.mean(data, axis=0)

    # 线性拟合
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = float(l.coef_.ravel()[0])
        b = float(l.intercept_.ravel()[0])
        if np.isclose(k, 0.0):
            raise ValueError(f"MSC failed for sample {i}: regression slope is zero.")
        msc[i, :] = (y - b) / k
    return msc.astype(np.float32)

# 小波变换
def wave(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after wave :(n_samples, n_features)
    """
    if pywt is None:
        raise ImportError("PyWavelets is required for wave preprocessing. Install package 'PyWavelets'.")
    data = _as_2d_float_array(data)
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(np.abs(coeffs[i])))
        datarec = pywt.waverec(coeffs, 'db8')
        if len(datarec) > len(data):
            datarec = datarec[:len(data)]
        elif len(datarec) < len(data):
            datarec = np.pad(datarec, (0, len(data) - len(datarec)), mode='edge')
        return datarec.astype(np.float32)

    return np.vstack([wave_(data[i]) for i in range(data.shape[0])]).astype(np.float32)

def Preprocessing(method, data):
    method_key = "none" if method is None else str(method).lower()
    methods = {
        "none": _as_2d_float_array,
        "mms": MMS,
        "ss": SS,
        "ct": CT,
        "snv": SNV,
        "ma": MA,
        "sg": SG,
        "msc": MSC,
        "d1": D1,
        "d2": D2,
        "dt": DT,
        "wave": wave,
    }
    if method_key not in methods:
        raise ValueError(
            f"Unsupported preprocessing method: {method}. "
            f"Supported methods: {list(SUPPORTED_PREPROCESSING_METHODS)}"
        )
    return methods[method_key](data)
