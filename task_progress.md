# 改进任务清单

- [x] 分析代码中与数据集特征列相关的错误
- [x] 第一次修正 PAPER_EXPECTED_SPECTRAL_LENGTH = 1556（基于用户描述）
- [x] 根据实际运行结果发现 CSV 实际只有 1555 个特征列
- [x] 修正 PAPER_EXPECTED_SPECTRAL_LENGTH = 1556 → 1555
- [x] 验证修改结果
- [x] 分析 LinearSVM 耗时原因并提供优化方案
- [x] 减少 LINEAR_SVM_C 候选参数 7→3
- [x] 开启特征选择 CLASSIFIER_USE_FEATURE_SELECTION = True, CLASSIFIER_MAX_FEATURES = 256
- [x] pipeline.py 中 cross_val_score 改为 n_jobs=-1 并行
- [x] 分析修改前后运行结果对比
- [x] 更新修改过程.txt 并推送到 GitHub
- [x] 根据运行结果分析模型瓶颈并提出改进方案
- [x] 实施改进方案一：提高 FUSION_DROPOUT 0.20 → 0.35
- [x] 实施改进方案二：降低 FEATURE_DIM 256 → 128
- [x] 实施改进方案三：增强光谱数据增强（噪声 std 0.005 → 0.015，幅度缩放 0.90-1.10）
- [x] 实施改进方案四：启用分支辅助损失（USE_BRANCH_AUX_LOSS = True）
- [x] 实施改进方案五：提高 Center Loss 权重 0.03 → 0.05，推迟启动到 epoch 10
- [ ] 更新修改过程.txt 并推送到 GitHub
