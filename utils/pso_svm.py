"""
PSO-SVM: Particle Swarm Optimization based Support Vector Machine

Faster implementation for small multimodal datasets:
- PSO searches C, gamma, and k inside a leakage-safe CV pipeline
- standardize and select features inside each CV fold
- use stratified CV with caching and early stopping
"""
from functools import partial

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class PSOSVM:
    def __init__(self, n_particles=15, max_iter=20,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0,
                 cv_folds=3, random_state=42, min_features=32, max_features_cap=256,
                 patience=6, scoring='accuracy', n_jobs=-1):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.min_features = min_features
        self.max_features_cap = max_features_cap
        self.patience = patience
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.bounds_C = (0.01, 100.0)
        self.bounds_gamma = (0.001, 1.0)

        self.best_C = None
        self.best_gamma = None
        self.best_k = None
        self.best_score = -np.inf
        self.best_feature_indices = None
        self.pipeline = None

        self._score_cache = {}

    def _build_pipeline(self, C, gamma, k):
        selector = SelectKBest(
            score_func=partial(mutual_info_classif, random_state=self.random_state),
            k=k,
        )
        svm = SVC(
            C=C,
            gamma=gamma,
            kernel='rbf',
            class_weight='balanced',
            random_state=self.random_state,
            decision_function_shape='ovr',
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('selector', selector),
            ('svm', svm),
        ])

    def _fitness(self, X, y, C, gamma, k, cv):
        key = (round(float(C), 6), round(float(gamma), 6), int(k))
        if key in self._score_cache:
            return self._score_cache[key]

        pipeline = self._build_pipeline(C, gamma, k)
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        score = float(scores.mean())

        self._score_cache[key] = score
        return score

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        class_counts = np.bincount(y)
        min_class_count = int(class_counts[class_counts > 0].min())
        if min_class_count < 2:
            raise ValueError(
                "PSO-SVM requires at least 2 samples in every class for stratified CV. "
                f"Class counts: {class_counts.tolist()}"
            )
        k_lower = max(1, min(self.min_features, n_features))
        k_upper = min(n_features, self.max_features_cap)
        if k_lower > k_upper:
            k_lower = k_upper

        cv = StratifiedKFold(
            n_splits=min(self.cv_folds, min_class_count),
            shuffle=True,
            random_state=self.random_state,
        )
        n_dims = 3

        positions = np.zeros((self.n_particles, n_dims))
        positions[:, 0] = rng.uniform(self.bounds_C[0], self.bounds_C[1], self.n_particles)
        positions[:, 1] = rng.uniform(self.bounds_gamma[0], self.bounds_gamma[1], self.n_particles)
        positions[:, 2] = rng.uniform(k_lower, k_upper, self.n_particles)
        velocities = rng.uniform(-1, 1, (self.n_particles, n_dims))

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.n_particles, -np.inf)
        global_best_position = positions[0].copy()
        global_best_score = -np.inf
        no_improve_iters = 0

        for iteration in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iter
            improved_this_iter = False

            for i in range(self.n_particles):
                C_i = np.clip(positions[i, 0], self.bounds_C[0], self.bounds_C[1])
                gamma_i = np.clip(positions[i, 1], self.bounds_gamma[0], self.bounds_gamma[1])
                k_i = int(round(np.clip(positions[i, 2], k_lower, k_upper)))

                score = self._fitness(X, y, C_i, gamma_i, k_i, cv)

                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()
                    improved_this_iter = True

            for i in range(self.n_particles):
                r1 = rng.random(n_dims)
                r2 = rng.random(n_dims)
                velocities[i] = (w * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                                 + self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]
                positions[i, 0] = np.clip(positions[i, 0], self.bounds_C[0], self.bounds_C[1])
                positions[i, 1] = np.clip(positions[i, 1], self.bounds_gamma[0], self.bounds_gamma[1])
                positions[i, 2] = np.clip(positions[i, 2], k_lower, k_upper)

            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"    PSO Iter {iteration+1}/{self.max_iter}: "
                      f"Best CV {self.scoring}={global_best_score:.4f}, "
                      f"C={global_best_position[0]:.4f}, "
                      f"gamma={global_best_position[1]:.4f}, "
                      f"k={int(round(global_best_position[2]))}")

            if improved_this_iter:
                no_improve_iters = 0
            else:
                no_improve_iters += 1
                if no_improve_iters >= self.patience:
                    print(f"  PSO early stop after {iteration+1} iterations (no improvement for {self.patience} rounds)")
                    break

        self.best_C = float(global_best_position[0])
        self.best_gamma = float(global_best_position[1])
        self.best_k = int(round(global_best_position[2]))
        self.best_k = max(k_lower, min(k_upper, self.best_k))
        self.best_score = global_best_score

        print("  PSO Optimization Complete:")
        print(f"    Best C={self.best_C:.4f}, Best gamma={self.best_gamma:.4f}, Best k={self.best_k}")
        print(f"    Best CV {self.scoring}={self.best_score:.4f}")

        self.pipeline = self._build_pipeline(self.best_C, self.best_gamma, self.best_k)
        self.pipeline.fit(X, y)
        self.best_feature_indices = self.pipeline.named_steps['selector'].get_support(indices=True)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)

    def decision_function(self, X):
        return self.pipeline.decision_function(X)
