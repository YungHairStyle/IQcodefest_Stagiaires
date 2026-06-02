import numpy as np
from dataclasses import dataclass
from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


@dataclass
class AnomalyDatasetConfig:
    dataset: str = "digits"
    normal_label: int = 0
    anomaly_labels: list | None = None

    n_train_normal: int = 120
    n_test_normal: int = 40
    n_test_anomaly: int = 120

    feature_dim: int = 64
    random_state: int = 7

    scale: str = "minmax"       # "minmax", "standard", or None
    use_pca: bool = True        # Useful for Fashion-MNIST and creditcard
    normalize_vectors: bool = True

    cache: bool = True          # Used by fetch_openml


class FlexibleAnomalyDataset:
    """
    General anomaly dataset builder.

    Supported datasets:
        "digits"         : sklearn built-in 8x8 digits
        "fashion_mnist"  : OpenML Fashion-MNIST, 28x28 images
        "creditcard"     : OpenML credit-card fraud dataset

    Returns:
        X_train_raw
        X_test_raw
        y_test
        raw
    """

    def __init__(self, config: AnomalyDatasetConfig):
        self.config = config

    def __call__(self):
        X, y, raw = self._load_dataset()

        y = np.asarray(y)

        if self.config.anomaly_labels is None:
            anomaly_labels = sorted(
                [label for label in np.unique(y) if label != self.config.normal_label]
            )
        else:
            anomaly_labels = self.config.anomaly_labels

        normal_idx = np.where(y == self.config.normal_label)[0]
        anomaly_idx = np.where(np.isin(y, anomaly_labels))[0]

        rng = np.random.default_rng(self.config.random_state)

        normal_idx = rng.permutation(normal_idx)
        anomaly_idx = rng.permutation(anomaly_idx)

        needed_normal = self.config.n_train_normal + self.config.n_test_normal
        needed_anomaly = self.config.n_test_anomaly

        if len(normal_idx) < needed_normal:
            raise ValueError(
                f"Not enough normal samples for label {self.config.normal_label}. "
                f"Need {needed_normal}, got {len(normal_idx)}."
            )

        if len(anomaly_idx) < needed_anomaly:
            raise ValueError(
                f"Not enough anomaly samples for labels {anomaly_labels}. "
                f"Need {needed_anomaly}, got {len(anomaly_idx)}."
            )

        train_normal_idx = normal_idx[: self.config.n_train_normal]

        test_normal_idx = normal_idx[
            self.config.n_train_normal :
            self.config.n_train_normal + self.config.n_test_normal
        ]

        test_anomaly_idx = anomaly_idx[: self.config.n_test_anomaly]

        test_indices = np.concatenate([test_normal_idx, test_anomaly_idx])

        X_train_raw = X[train_normal_idx]
        X_test_raw = X[test_indices]

        y_test = np.concatenate(
            [
                np.zeros(len(test_normal_idx), dtype=int),
                np.ones(len(test_anomaly_idx), dtype=int),
            ]
        )

        X_train_processed, X_test_processed, preprocess_info = self._preprocess(
            X_train_raw,
            X_test_raw,
        )

        raw.update(
            {
                "dataset": self.config.dataset,
                "normal_label": self.config.normal_label,
                "anomaly_labels": anomaly_labels,
                "train_indices": train_normal_idx,
                "test_indices": test_indices,
                "test_labels_original": y[test_indices],
                "preprocess_info": preprocess_info,
            }
        )

        return X_train_processed, X_test_processed, y_test, raw

    def _load_dataset(self):
        dataset = self.config.dataset.lower()

        if dataset == "digits":
            digits = load_digits()

            X = digits.data.astype(float) / 16.0
            y = digits.target.astype(int)

            raw = {
                "images": digits.images.astype(float) / 16.0,
                "feature_shape": (8, 8),
                "original_dim": 64,
            }

            return X, y, raw

        elif dataset == "fashion_mnist":
            data = fetch_openml(
                data_id=40996,
                as_frame=False,
                cache=self.config.cache,
            )

            X = data.data.astype(float) / 255.0
            y = data.target.astype(int)

            raw = {
                "images": X.reshape(-1, 28, 28),
                "feature_shape": (28, 28),
                "original_dim": 784,
                "class_names": {
                    0: "T-shirt/top",
                    1: "Trouser",
                    2: "Pullover",
                    3: "Dress",
                    4: "Coat",
                    5: "Sandal",
                    6: "Shirt",
                    7: "Sneaker",
                    8: "Bag",
                    9: "Ankle boot",
                },
            }

            return X, y, raw

        elif dataset == "creditcard":
            data = fetch_openml(
                data_id=1597,
                as_frame=False,
                cache=self.config.cache,
            )

            X = data.data.astype(float)

            # OpenML targets often come back as strings like "0", "1".
            y = data.target.astype(int)

            raw = {
                "images": None,
                "feature_shape": None,
                "original_dim": X.shape[1],
                "class_names": {
                    0: "normal",
                    1: "fraud",
                },
            }

            return X, y, raw

        else:
            raise ValueError(
                f"Unknown dataset '{self.config.dataset}'. "
                "Use 'digits', 'fashion_mnist', or 'creditcard'."
            )

    def _preprocess(self, X_train, X_test):
        info = {}

        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)

        if self.config.scale == "minmax":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            info["scale"] = "minmax"

        elif self.config.scale == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            info["scale"] = "standard"

        elif self.config.scale is None:
            info["scale"] = None

        else:
            raise ValueError("scale must be 'minmax', 'standard', or None.")

        if self.config.use_pca:
            if self.config.feature_dim is None:
                raise ValueError("feature_dim must be set when use_pca=True.")

            pca = PCA(
                n_components=self.config.feature_dim,
                random_state=self.config.random_state,
            )

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            info["pca"] = True
            info["feature_dim"] = self.config.feature_dim
            info["explained_variance_ratio_sum"] = float(
                np.sum(pca.explained_variance_ratio_)
            )

        else:
            info["pca"] = False
            info["feature_dim"] = X_train.shape[1]

        if self.config.normalize_vectors:
            X_train = self._l2_normalize(X_train)
            X_test = self._l2_normalize(X_test)
            info["normalize_vectors"] = True
        else:
            info["normalize_vectors"] = False

        self._check_power_of_two(X_train.shape[1])

        return X_train, X_test, info

    def _l2_normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)

        # Avoid division by zero.
        norms[norms == 0.0] = 1.0

        return X / norms

    def _check_power_of_two(self, dim):
        n_qubits_float = np.log2(dim)
        n_qubits = int(round(n_qubits_float))

        if 2**n_qubits != dim:
            raise ValueError(
                f"Final feature dimension must be a power of two for amplitude encoding. "
                f"Got feature_dim={dim}."
            )

        return n_qubits

    def get_info(self):
        return self.config.__dict__