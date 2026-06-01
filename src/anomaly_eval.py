import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


class AnomalyEvaluator:
    """
    Evaluate anomaly detection scores.

    Supported threshold strategies:

        "expected_fraction":
            Choose the threshold so that the predicted anomaly fraction matches
            the true anomaly fraction in y_true.

            This is convenient for a controlled benchmark.

        "percentile_75":
            Flag the top 25% highest scores as anomalies.

        "zero":
            Use threshold = 0.

        "best_f1":
            Sweep over thresholds and choose the one with best F1.
            This is optimistic because it uses test labels. Use mainly for
            analysis, not as the main deployment claim.
    """

    def __init__(self, threshold_strategy="expected_fraction"):
        """
        Parameters
        ----------
        threshold_strategy : str
            Strategy used to convert anomaly scores into labels.
        """
        allowed = ["expected_fraction", "percentile_75", "zero", "best_f1"]

        if threshold_strategy not in allowed:
            raise ValueError(
                f"Unknown threshold_strategy={threshold_strategy}. "
                f"Allowed values are {allowed}."
            )

        self.threshold_strategy = threshold_strategy

    def choose_threshold(self, y_true, scores):
        """
        Choose a threshold for anomaly scores.

        Parameters
        ----------
        y_true : np.ndarray
            0 = normal, 1 = anomaly

        scores : np.ndarray
            Larger = more anomalous

        Returns
        -------
        threshold : float
        """
        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(scores, dtype=float)

        if self.threshold_strategy == "expected_fraction":
            # If 75% of test examples are anomalies, choose threshold so that
            # top 75% of scores are predicted anomalies.
            anomaly_fraction = np.mean(y_true)
            percentile = 100 * (1 - anomaly_fraction)
            return float(np.percentile(scores, percentile))

        if self.threshold_strategy == "percentile_75":
            # Top 25% most anomalous are predicted anomalies.
            return float(np.percentile(scores, 75))

        if self.threshold_strategy == "zero":
            return 0.0

        if self.threshold_strategy == "best_f1":
            # Optimistic threshold sweep.
            unique_scores = np.unique(scores)

            best_threshold = unique_scores[0]
            best_f1 = -1.0

            for threshold in unique_scores:
                y_pred = (scores >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            return float(best_threshold)

        raise RuntimeError("Unreachable threshold strategy branch.")

    def evaluate(self, y_true, scores):
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            0 = normal, 1 = anomaly

        scores : np.ndarray
            Larger = more anomalous

        Returns
        -------
        metrics : dict
        """
        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(scores, dtype=float)

        threshold = self.choose_threshold(y_true, scores)
        y_pred = (scores >= threshold).astype(int)

        auc = roc_auc_score(y_true, scores)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        report = classification_report(
            y_true,
            y_pred,
            target_names=["normal", "anomaly"],
            zero_division=0,
        )

        return {
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": float(threshold),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
            "scores": scores,
            "threshold_strategy": self.threshold_strategy,
        }

    def get_info(self):
        """
        Return metadata.
        """
        return {
            "type": "anomaly_evaluator",
            "threshold_strategy": self.threshold_strategy,
        }