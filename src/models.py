import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSVMAnomalyModel:
    """
    One-class SVM anomaly detector using precomputed kernels.

    Training:
        model.fit(K_train)

    Scoring:
        scores = model.score(K_test)

    By convention in this project:

        larger score = more anomalous

    sklearn's OneClassSVM decision_function uses the opposite convention:
        larger decision value = more normal/inlier-like

    So we flip the sign internally.
    """

    def __init__(self, nu=0.1):
        """
        Parameters
        ----------
        nu : float
            OneClassSVM parameter.

            Roughly:
                - upper bound on fraction of training errors
                - lower bound on fraction of support vectors

            For anomaly detection, common values are 0.05, 0.1, 0.2.
        """
        self.nu = nu
        self.model = OneClassSVM(kernel="precomputed", nu=nu)
        self.is_fitted_ = False

    def fit(self, K_train):
        """
        Fit the one-class SVM on the training kernel matrix.

        Parameters
        ----------
        K_train : np.ndarray
            Shape: (n_train, n_train)

        Returns
        -------
        self
        """
        K_train = np.asarray(K_train, dtype=float)

        if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
            raise ValueError(
                "K_train must be a square matrix of shape (n_train, n_train)."
            )

        self.model.fit(K_train)
        self.is_fitted_ = True

        return self

    def score(self, K_test):
        """
        Compute anomaly scores for test samples.

        Parameters
        ----------
        K_test : np.ndarray
            Shape: (n_test, n_train)

        Returns
        -------
        scores : np.ndarray
            Shape: (n_test,)
            Larger score means more anomalous.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before score().")

        K_test = np.asarray(K_test, dtype=float)

        # sklearn decision_function:
        # larger = more normal
        #
        # We flip the sign:
        # larger = more anomalous
        scores = -self.model.decision_function(K_test)

        return scores

    def predict_from_scores(self, scores, threshold):
        """
        Convert anomaly scores into binary predictions.

        Parameters
        ----------
        scores : np.ndarray
            Larger means more anomalous.

        threshold : float
            Samples with score >= threshold are predicted as anomalies.

        Returns
        -------
        y_pred : np.ndarray
            0 = normal
            1 = anomaly
        """
        scores = np.asarray(scores)
        return (scores >= threshold).astype(int)

    def get_info(self):
        """
        Return metadata.
        """
        return {
            "type": "one_class_svm",
            "kernel": "precomputed",
            "nu": self.nu,
            "is_fitted": self.is_fitted_,
        }