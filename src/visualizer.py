import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay


class AnomalyVisualizer:
    """
    Visualization helper for image anomaly detection experiments.
    """

    def plot_images(self, images, labels=None, n=16, title=None, cols=8):
        """
        Plot a grid of 8x8 digit images.

        Parameters
        ----------
        images : np.ndarray
            Shape: (n_images, 8, 8)

        labels : array-like or None
            Optional labels to show above each image.

        n : int
            Maximum number of images to show.

        title : str or None
            Figure title.

        cols : int
            Number of columns in the grid.
        """
        n = min(n, len(images))
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(1.4 * cols, 1.4 * rows))

        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis("off")

            if labels is not None:
                plt.title(str(labels[i]), fontsize=9)

        if title is not None:
            plt.suptitle(title)

        plt.tight_layout()
        plt.show()

    def plot_kernel(self, K, title="Kernel matrix"):
        """
        Plot a kernel matrix as a heatmap.
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(K, aspect="auto")
        plt.colorbar(label="kernel value")
        plt.title(title)
        plt.xlabel("index")
        plt.ylabel("index")
        plt.tight_layout()
        plt.show()

    def plot_roc(self, y_true, scores, title="ROC curve"):
        """
        Plot ROC curve.
        """
        RocCurveDisplay.from_predictions(y_true, scores)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion matrix"):
        """
        Plot confusion matrix.
        """
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=["normal", "anomaly"],
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def show_most_anomalous(
        self,
        images,
        labels,
        scores,
        top_k=16,
        title="Most anomalous images",
    ):
        """
        Show images with largest anomaly scores.
        """
        scores = np.asarray(scores)
        idx = np.argsort(scores)[-top_k:][::-1]

        self.plot_images(
            images[idx],
            labels=np.asarray(labels)[idx],
            n=top_k,
            title=title,
        )

    def show_most_normal(
        self,
        images,
        labels,
        scores,
        top_k=16,
        title="Most normal-looking images",
    ):
        """
        Show images with smallest anomaly scores.
        """
        scores = np.asarray(scores)
        idx = np.argsort(scores)[:top_k]

        self.plot_images(
            images[idx],
            labels=np.asarray(labels)[idx],
            n=top_k,
            title=title,
        )