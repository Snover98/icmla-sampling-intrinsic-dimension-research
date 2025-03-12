from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection


def sample_dataset_random(x: np.ndarray[float],
                          n_samples: int, random_seed: int = None) -> np.ndarray[int]:
    rng = np.random
    if random_seed:
        rng = np.random.default_rng(random_seed)

    choice_idx = rng.choice(np.arange(len(x)), size=n_samples, replace=False).astype(int)

    return choice_idx


def sample_dataset_jls_kmeans(x: np.ndarray[float],
                              n_samples: int, jls_dim: int,
                              random_seed: int = None) -> np.ndarray[int]:
    train_x_transformed = SparseRandomProjection(jls_dim, random_state=random_seed).fit_transform(x.reshape(len(x), -1))

    kmeans_labels = KMeans(n_samples, random_state=random_seed).fit_predict(train_x_transformed)
    indices_per_cluster: dict[Any, list[int]] = {
        c_label: [idx for idx in range(len(kmeans_labels)) if kmeans_labels[idx] == c_label]
        for c_label in kmeans_labels
    }

    rng = np.random
    if random_seed:
        rng = np.random.default_rng(random_seed)

    return np.array([rng.choice(cluster_indices) for cluster_indices in indices_per_cluster.values()]).astype(int)
