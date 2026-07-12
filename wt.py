import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, concatenate
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from imblearn.combine import SMOTETomek
import pywt  # 导入小波变换库

# 读取CSV文件
data = pd.read_csv('产地-7.csv')

# 分离特征和标签
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 标签二值化
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

# 获取类别名称，确保它们是字符串
target_names = label_binarizer.classes_.astype(str)


# D1: 一阶微分
def first_derivative(x):
    return np.diff(x, n=1, axis=-1)


# SG: Savitzky-Golay滤波
def savitzky_golay_filtering(x, window_length=5, polyorder=2):
    if x.shape[-1] < window_length:
        window_length = x.shape[-1] - 1 if x.shape[-1] % 2 == 0 else x.shape[-1]
        if window_length < 3:
            return x  # 如果窗口太小，直接返回原数据
    return savgol_filter(x, window_length, polyorder, mode='nearest', axis=-1)


# SNV: 标准正态变量变换
def snv(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    std_dev = np.std(x, axis=-1, ddof=1, keepdims=True)
    # 避免除以零
    std_dev[std_dev == 0] = 1e-6
    return (x - mean) / std_dev


# 小波变换特征提取
def wavelet_transform(x, wavelet='db4', level=3):
    """对每个样本进行小波变换，提取近似系数作为特征"""
    transformed = []
    for sample in x:
        # 确保样本长度足够
        max_level = pywt.dwt_max_level(len(sample), wavelet)
        actual_level = min(level, max_level) if max_level > 0 else 1

        # 进行小波分解
        coeffs = pywt.wavedec(sample, wavelet, level=actual_level)
        # 提取近似系数（低频部分）
        cA = coeffs[0]

        # 如果长度不匹配，进行插值
        if len(cA) < len(sample):
            cA = np.interp(np.linspace(0, len(cA) - 1, len(sample)),
                           np.arange(len(cA)), cA)
        transformed.append(cA)
    return np.array(transformed)



X_d1 = first_derivative(X)
X_sg = savitzky_golay_filtering(X_d1)
X_snv = snv(X_sg)

# 确保所有样本具有相同的长度
min_length = min([len(sample) for sample in X_snv])
X_original = np.array([sample[:min_length] for sample in X_snv])


X_wavelet = wavelet_transform(X, level=3)
X_wavelet_d1 = first_derivative(X_wavelet)
X_wavelet_sg = savitzky_golay_filtering(X_wavelet_d1)
X_wavelet_snv = snv(X_wavelet_sg)

# 确保小波特征与原始特征长度一致
X_wavelet_processed = np.array([sample[:min_length] for sample in X_wavelet_snv])


# 重塑特征为3D格式（样本数，特征长度，通道数=1）
X_original_reshaped = X_original.reshape((X_original.shape[0], X_original.shape[1], 1))
X_wavelet_reshaped = X_wavelet_processed.reshape((X_wavelet_processed.shape[0], X_wavelet_processed.shape[1], 1))

# 划分训练集和测试集
X_original_train, X_original_test, X_wavelet_train, X_wavelet_test, y_train, y_test = train_test_split(
    X_original_reshaped, X_wavelet_reshaped, y_encoded, test_size=0.2, random_state=42)#stratify=y


# 原始特征标准化
scaler_original = StandardScaler()
# 拟合训练集→转换训练集
X_original_train_flat = X_original_train.reshape(X_original_train.shape[0], -1)
X_original_train_scaled_flat = scaler_original.fit_transform(X_original_train_flat)
X_original_train_scaled = X_original_train_scaled_flat.reshape(X_original_train.shape)
# 用训练集的标准化器转换测试集
X_original_test_flat = X_original_test.reshape(X_original_test.shape[0], -1)
X_original_test_scaled = scaler_original.transform(X_original_test_flat).reshape(X_original_test.shape)

# 小波特征标准化（仅用训练集拟合）
scaler_wavelet = StandardScaler()
X_wavelet_train_flat = X_wavelet_train.reshape(X_wavelet_train.shape[0], -1)
X_wavelet_train_scaled_flat = scaler_wavelet.fit_transform(X_wavelet_train_flat)
X_wavelet_train_scaled = X_wavelet_train_scaled_flat.reshape(X_wavelet_train.shape)

X_wavelet_test_flat = X_wavelet_test.reshape(X_wavelet_test.shape[0], -1)
X_wavelet_test_scaled = scaler_wavelet.transform(X_wavelet_test_flat).reshape(X_wavelet_test.shape)

# 处理类别不平衡（仅对训练集进行）
smote_tomek = SMOTETomek(random_state=42)

# 合并训练集的原始特征和小波特征（展平后过采样）
X_combined_train_flat = np.hstack((X_original_train_scaled_flat, X_wavelet_train_scaled_flat))
X_combined_balanced_flat, y_train_balanced = smote_tomek.fit_resample(
    X_combined_train_flat, y_train.argmax(axis=1))  # y_train为独热编码，需转为整数标签

# 分离平衡后的特征并重塑为3D
split_idx = X_original_train_scaled_flat.shape[1]  # 原始特征的维度
X_original_balanced_flat = X_combined_balanced_flat[:, :split_idx]
X_wavelet_balanced_flat = X_combined_balanced_flat[:, split_idx:]

X_original_balanced = X_original_balanced_flat.reshape(
    X_original_balanced_flat.shape[0], X_original_train.shape[1], 1)
X_wavelet_balanced = X_wavelet_balanced_flat.reshape(
    X_wavelet_balanced_flat.shape[0], X_wavelet_train.shape[1], 1)

# 转换标签回独热编码形式
y_train_balanced = to_categorical(y_train_balanced)

# ===== 构建双输入CNN模型 =====
# 原始特征输入分支
input_original = Input(shape=(X_original_balanced.shape[1], 1))
x_original = Conv1D(filters=32, kernel_size=3, activation='relu',
                    kernel_regularizer=l2(0.01))(input_original)
x_original = MaxPooling1D(pool_size=2)(x_original)
x_original = Dropout(0.3)(x_original)
x_original = Conv1D(filters=64, kernel_size=3, activation='relu',
                    kernel_regularizer=l2(0.01))(x_original)
x_original = MaxPooling1D(pool_size=2)(x_original)
x_original = Flatten()(x_original)

# 小波特征输入分支
input_wavelet = Input(shape=(X_wavelet_balanced.shape[1], 1))
x_wavelet = Conv1D(filters=32, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(0.01))(input_wavelet)
x_wavelet = MaxPooling1D(pool_size=2)(x_wavelet)
x_wavelet = Dropout(0.3)(x_wavelet)
x_wavelet = Conv1D(filters=64, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(0.01))(x_wavelet)
x_wavelet = MaxPooling1D(pool_size=2)(x_wavelet)
x_wavelet = Flatten()(x_wavelet)

# 合并两个分支
merged = concatenate([x_original, x_wavelet])
merged = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
merged = Dropout(0.5)(merged)
outputs = Dense(len(target_names), activation='softmax')(merged)

model = Model(inputs=[input_original, input_wavelet], outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型结构
model.summary()

# 定义回调函数 - 现在使用测试集作为验证集
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('best_model_wavelet.h5', monitor='val_loss',
                                   save_best_only=True, mode='min')

# 训练模型
history = model.fit(
    [X_original_balanced, X_wavelet_balanced], y_train_balanced,
    epochs=120,#120
    batch_size=64,
    validation_data=([X_original_test_scaled, X_wavelet_test_scaled], y_test),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# 加载最佳模型权重
model.load_weights('best_model_wavelet.h5')

# 评估模型
y_pred = model.predict([X_original_test_scaled, X_wavelet_test_scaled])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 计算各种指标
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# 打印指标
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision, weighted): {precision:.4f}")
print(f"召回率 (Recall, weighted): {recall:.4f}")
print(f"F1分数 (F1-score, weighted): {f1:.4f}")
print()

# 打印混淆矩阵和分类报告
print("混淆矩阵:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# 绘制混淆矩阵热图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('confusion_matrix_wavelet.png')
plt.show()

print("\n分类报告:")
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy') 
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves_wavelet.png')
plt.show()