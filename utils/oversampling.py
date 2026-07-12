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
    def __init__(self, k_neighbors=5, k_clusters=8, random_state=42):
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if k_clusters < 1:
            raise ValueError(f"k_clusters must be >= 1, got {k_clusters}")
        self.k_neighbors = k_neighbors
        self.k_clusters = k_clusters
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
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
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_minority)

            cluster_sizes = np.bincount(labels, minlength=n_clusters)
            total_minority = len(X_minority)
            eligible_clusters = [
                cluster_idx for cluster_idx in range(n_clusters)
                if cluster_sizes[cluster_idx] >= 2
            ]
            if not eligible_clusters:
                raise RuntimeError(
                    f"KMeans-SMOTE cannot generate samples for class {cls}: "
                    "all clusters contain fewer than 2 samples."
                )

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
