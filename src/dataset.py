import numpy as np
from sklearn.datasets import load_digits


class DigitAnomalyDataset:
    """
    Build an anomaly detection dataset from sklearn's built-in digits dataset.

    This class creates the following split:

        X_train_raw:
            Contains only normal images.

        X_test_raw:
            Contains normal images and anomaly images.

        y_test:
            Binary labels for the test set.
            0 = normal
            1 = anomaly

        raw:
            A metadata dictionary containing original 8x8 images and labels
            for visualization.
    """

    def __init__(
        self,
        normal_digit=0,
        anomaly_digits=None,
        n_train_normal=120,
        n_test_normal=40,
        n_test_anomaly=120,
        random_state=7,
    ):
        """
        Parameters
        ----------
        normal_digit : int
            The digit treated as the normal class.

        anomaly_digits : list[int] or None
            Digits treated as anomalies.
            If None, all digits except normal_digit are anomalies.

        n_train_normal : int
            Number of normal examples used for training.

        n_test_normal : int
            Number of normal examples used for testing.

        n_test_anomaly : int
            Number of anomaly examples used for testing.

        random_state : int
            Random seed for reproducible splitting.
        """
        self.normal_digit = normal_digit

        if anomaly_digits is None:
            anomaly_digits = [d for d in range(10) if d != normal_digit]

        self.anomaly_digits = anomaly_digits
        self.n_train_normal = n_train_normal
        self.n_test_normal = n_test_normal
        self.n_test_anomaly = n_test_anomaly
        self.random_state = random_state

    def __call__(self):
        """
        Load and split the digit dataset.

        Returns
        -------
        X_train_raw : np.ndarray
            Shape: (n_train_normal, 64)
            Flattened normalized normal training images.

        X_test_raw : np.ndarray
            Shape: (n_test_normal + n_test_anomaly, 64)
            Flattened normalized test images.

        y_test : np.ndarray
            Shape: (n_test_normal + n_test_anomaly,)
            Binary test labels.
            0 = normal
            1 = anomaly

        raw : dict
            Dictionary containing original images and labels for plotting.
        """
        digits = load_digits()

        # digits.data has shape (n_samples, 64).
        # Pixel values are in [0, 16], so divide by 16 to put them in [0, 1].
        X = digits.data.astype(float) / 16.0

        # digits.images has shape (n_samples, 8, 8), useful for plotting.
        images = digits.images.astype(float) / 16.0

        # Original labels are integers 0-9.
        y = digits.target

        # Indices for normal examples and anomaly examples.
        normal_idx = np.where(y == self.normal_digit)[0]
        anomaly_idx = np.where(np.isin(y, self.anomaly_digits))[0]

        rng = np.random.default_rng(self.random_state)

        # Shuffle so the split is random but reproducible. 
        normal_idx = rng.permutation(normal_idx)
        anomaly_idx = rng.permutation(anomaly_idx)

        # Select normal training examples.
        train_normal_idx = normal_idx[: self.n_train_normal]

        # Select normal test examples.
        test_normal_idx = normal_idx[
            self.n_train_normal : self.n_train_normal + self.n_test_normal
        ]

        # Select anomaly test examples.
        test_anomaly_idx = anomaly_idx[: self.n_test_anomaly]

        # Build train and test arrays.
        X_train_raw = X[train_normal_idx]

        test_indices = np.concatenate([test_normal_idx, test_anomaly_idx])
        X_test_raw = X[test_indices]

        # y_test is binary:
        # 0 for normal, 1 for anomaly.
        y_test = np.concatenate(
            [
                np.zeros(len(test_normal_idx), dtype=int),
                np.ones(len(test_anomaly_idx), dtype=int),
            ]
        )

        raw = {
            "train_images": images[train_normal_idx],
            "test_images": images[test_indices],
            "test_labels_original": y[test_indices],
            "normal_digit": self.normal_digit,
            "anomaly_digits": self.anomaly_digits,
            "train_indices": train_normal_idx,
            "test_indices": test_indices,
        }

        return X_train_raw, X_test_raw, y_test, raw

    def get_info(self):
        """
        Return a dictionary describing the dataset configuration.
        """
        return {
            "dataset": "sklearn_digits",
            "normal_digit": self.normal_digit,
            "anomaly_digits": self.anomaly_digits,
            "n_train_normal": self.n_train_normal,
            "n_test_normal": self.n_test_normal,
            "n_test_anomaly": self.n_test_anomaly,
            "random_state": self.random_state,
        }
    