"""
KMeans-SMOTE Oversampling Module

Paper: "Multimodal feature fusion network based on image and NIR spectroscopy
        for diesel fuel brand identification"

Algorithm:
1. Use k-means to cluster minority class samples into k_clusters sub-clusters
2. Select clusters to oversample and determine number of samples per cluster
3. Apply SMOTE within each cluster to generate synthetic samples
   x_new = x + rand(0,1) * (x_n - x)
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class KMeansSMOTE:
    def __init__(self, k_neighbors=5, k_clusters=8, random_state=42, strategy="max"):
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if k_clusters < 1:
            raise ValueError(f"k_clusters must be >= 1, got {k_clusters}")
        if strategy not in ("max", "median"):
            raise ValueError(f"strategy must be 'max' or 'median', got {strategy}")
        self.k_neighbors = k_neighbors
        self.k_clusters = k_clusters
        self.random_state = random_state
        self.strategy = strategy

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.cluster_counts_ = {}
        classes, counts = np.unique(y, return_counts=True)
        if self.strategy == "median":
            max_count = int(np.median(counts))
        else:
            max_count = counts.max()
        majority_class = classes[counts.argmax()]

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for cls in classes:
            if cls == majority_class:
                continue
            X_minority = X[y == cls]
            n_minority = len(X_minority)
            n_to_generate = max_count - n_minority

            if n_to_generate <= 0:
                continue

            n_clusters = min(self.k_clusters, n_minority)

            def cluster_with(current_clusters):
                model = KMeans(
                    n_clusters=current_clusters,
                    random_state=self.random_state,
                    n_init=10,
                )
                current_labels = model.fit_predict(X_minority)
                current_sizes = np.bincount(current_labels, minlength=current_clusters)
                current_eligible = [
                    cluster_idx for cluster_idx in range(current_clusters)
                    if current_sizes[cluster_idx] >= 2
                ]
                return current_labels, current_sizes, current_eligible

            labels, cluster_sizes, eligible_clusters = cluster_with(n_clusters)
            configured_clusters = n_clusters

            # Preserve the configured clustering whenever it works. Only a class
            # made entirely of singleton clusters enters this adaptive fallback.
            if not eligible_clusters:
                n_clusters = min(self.k_clusters, max(1, n_minority // 2))
                while n_clusters >= 1:
                    labels, cluster_sizes, eligible_clusters = cluster_with(n_clusters)
                    if eligible_clusters:
                        break
                    n_clusters -= 1
                if eligible_clusters:
                    print(
                        f"  Class {int(cls)}: KMeans-SMOTE reduced clusters "
                        f"from {configured_clusters} to {n_clusters} to avoid singleton-only clusters."
                    )

            if not eligible_clusters:
                raise RuntimeError(
                    f"KMeans-SMOTE cannot generate samples for class {cls}: "
                    "the class contains fewer than 2 usable samples."
                )
            self.cluster_counts_[int(cls)] = int(n_clusters)

            eligible_counts = np.array([cluster_sizes[idx] for idx in eligible_clusters], dtype=np.float64)
            proportions = eligible_counts / eligible_counts.sum()
            raw_allocations = proportions * n_to_generate
            allocations = np.floor(raw_allocations).astype(int)
            remainder = int(n_to_generate - allocations.sum())
            if remainder > 0:
                order = np.argsort(-(raw_allocations - allocations))
                for pos in order[:remainder]:
                    allocations[pos] += 1

            for cluster_idx, n_gen_cluster in zip(eligible_clusters, allocations):
                cluster_mask = labels == cluster_idx
                X_cluster = X_minority[cluster_mask]
                n_cluster = len(X_cluster)
                if n_gen_cluster == 0:
                    continue

                k_nn = min(self.k_neighbors + 1, n_cluster)
                nn = NearestNeighbors(n_neighbors=k_nn, metric='euclidean')
                nn.fit(X_cluster)

                synthetic_samples = []
                for _ in range(n_gen_cluster):
                    idx = rng.randint(0, n_cluster)
                    sample = X_cluster[idx]
                    _, indices = nn.kneighbors(sample.reshape(1, -1))
                    candidate_indices = indices[0][indices[0] != idx]
                    if candidate_indices.size == 0:
                        raise RuntimeError(
                            f"KMeans-SMOTE could not find a non-self neighbor for class {cls}."
                        )
                    neighbor_idx = rng.choice(candidate_indices)
                    neighbor = X_cluster[neighbor_idx]
                    diff = neighbor - sample
                    gap = rng.uniform(0, 1)
                    new_sample = sample + gap * diff
                    synthetic_samples.append(new_sample)

                if synthetic_samples:
                    X_resampled.append(np.array(synthetic_samples))
                    y_resampled.append(np.full(len(synthetic_samples), cls))

        X_out = np.vstack(X_resampled)
        y_out = np.concatenate(y_resampled)
        return X_out, y_out
