import numpy as np


class QuantumAnomalyExperiment:
    """
    Main class that runs the anomaly detection experiment.

    This class is intentionally high-level. It does not implement PCA,
    amplitude encoding, quantum circuits, kernels, or SVM internals.

    It just coordinates the components.

    Required component interfaces:

        dataset.load()
            returns X_train_raw, X_test_raw, y_test, raw

        encoder.fit_transform(X_train_raw)
            returns X_train_encoded

        encoder.transform(X_test_raw)
            returns X_test_encoded

        kernel.fit_transform(X_train_encoded)
            returns K_train

        kernel.transform(X_test_encoded)
            returns K_test

        model.fit(K_train)

        model.score(K_test)
            returns anomaly scores

        evaluator.evaluate(y_test, scores)
            returns metrics
    """

    def __init__(
        self,
        dataset,
        encoder,
        kernel,
        model,
        evaluator,
        name="experiment",
        visualizer=None,
    ):
        """
        Parameters
        ----------
        dataset : object
            Dataset object with load() method.

        encoder : object
            Encoder object with fit_transform() and transform().

        kernel : object
            Kernel object with fit_transform() and transform().

        model : object
            Anomaly model with fit() and score().

        evaluator : object
            Evaluator with evaluate().

        name : str
            Experiment name.

        visualizer : object or None
            Optional visualizer object.
        """
        self.dataset = dataset
        self.encoder = encoder
        self.kernel = kernel
        self.model = model
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.name = name

        self.results_ = None

    def run(self, verbose=True):
        """
        Run the full experiment.

        Returns
        -------
        results : dict
            Dictionary containing data, kernels, scores, metrics, and metadata.
        """
        if verbose:
            print(f"Running experiment: {self.name}")

        # ------------------------------------------------------------
        # 1. Load raw data.
        # ------------------------------------------------------------
        X_train_raw, X_test_raw, y_test, raw = self.dataset.load()

        if verbose:
            print("Loaded data:")
            print("  X_train_raw:", X_train_raw.shape)
            print("  X_test_raw:", X_test_raw.shape)
            print("  y_test:", y_test.shape)

        # ------------------------------------------------------------
        # 2. Encode raw data.
        # ------------------------------------------------------------
        X_train_encoded = self.encoder.fit_transform(X_train_raw)
        X_test_encoded = self.encoder.transform(X_test_raw)

        if verbose:
            print("Encoded data:")
            print("  X_train_encoded:", X_train_encoded.shape)
            print("  X_test_encoded:", X_test_encoded.shape)

        # ------------------------------------------------------------
        # 3. Compute training kernel.
        # ------------------------------------------------------------
        K_train = self.kernel.fit_transform(X_train_encoded)

        if verbose:
            print("Training kernel:")
            print("  K_train:", K_train.shape)

        # ------------------------------------------------------------
        # 4. Fit anomaly model.
        # ------------------------------------------------------------
        self.model.fit(K_train)

        if verbose:
            print("Model fitted.")

        # ------------------------------------------------------------
        # 5. Compute test-vs-train kernel.
        # ------------------------------------------------------------
        K_test = self.kernel.transform(X_test_encoded)

        if verbose:
            print("Test kernel:")
            print("  K_test:", K_test.shape)

        # ------------------------------------------------------------
        # 6. Compute anomaly scores.
        # ------------------------------------------------------------
        scores = self.model.score(K_test)

        if verbose:
            print("Scores:")
            print("  scores:", scores.shape)

        # ------------------------------------------------------------
        # 7. Evaluate.
        # ------------------------------------------------------------
        metrics = self.evaluator.evaluate(y_test, scores)

        if verbose:
            print("Metrics:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  F1:  {metrics['f1']:.4f}")

        # ------------------------------------------------------------
        # 8. Package all results.
        # ------------------------------------------------------------
        results = {
            "name": self.name,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "X_train_encoded": X_train_encoded,
            "X_test_encoded": X_test_encoded,
            "y_test": y_test,
            "raw": raw,
            "K_train": K_train,
            "K_test": K_test,
            "scores": scores,
            "metrics": metrics,
            "y_pred": metrics["y_pred"],
            "dataset_info": self.dataset.get_info()
            if hasattr(self.dataset, "get_info")
            else {},
            "encoder_info": self.encoder.get_info()
            if hasattr(self.encoder, "get_info")
            else {},
            "kernel_info": self.kernel.get_info()
            if hasattr(self.kernel, "get_info")
            else {},
            "model_info": self.model.get_info()
            if hasattr(self.model, "get_info")
            else {},
            "evaluator_info": self.evaluator.get_info()
            if hasattr(self.evaluator, "get_info")
            else {},
        }

        self.results_ = results
        return results

    def summarize(self):
        """
        Print a compact summary after run().
        """
        if self.results_ is None:
            raise RuntimeError("No results yet. Call run() first.")

        metrics = self.results_["metrics"]

        print(f"Experiment: {self.name}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print()
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])
        print()
        print("Classification report:")
        print(metrics["classification_report"])

    def sanity_checks(self):
        """
        Run useful sanity checks on the result.

        Returns
        -------
        checks : dict
        """
        if self.results_ is None:
            raise RuntimeError("No results yet. Call run() first.")

        K_train = self.results_["K_train"]
        scores = self.results_["scores"]
        y_test = self.results_["y_test"]

        checks = {
            "K_train_shape": K_train.shape,
            "K_train_min": float(np.min(K_train)),
            "K_train_max": float(np.max(K_train)),
            "K_train_has_nan": bool(np.isnan(K_train).any()),
            "K_train_symmetric": bool(np.allclose(K_train, K_train.T)),
            "K_train_diag_close_to_one": bool(np.allclose(np.diag(K_train), 1.0)),
            "normal_score_mean": float(np.mean(scores[y_test == 0])),
            "anomaly_score_mean": float(np.mean(scores[y_test == 1])),
        }

        return checks

    def plot_results(self):
        """
        Plot standard result figures using the visualizer.

        Requires visualizer to be provided.
        """
        if self.visualizer is None:
            raise RuntimeError("No visualizer was provided.")

        if self.results_ is None:
            raise RuntimeError("No results yet. Call run() first.")

        raw = self.results_["raw"]
        y_test = self.results_["y_test"]
        scores = self.results_["scores"]
        y_pred = self.results_["y_pred"]
        K_train = self.results_["K_train"]

        self.visualizer.plot_kernel(
            K_train,
            title=f"{self.name}: training kernel",
        )

        self.visualizer.plot_roc(
            y_test,
            scores,
            title=f"{self.name}: ROC curve",
        )

        self.visualizer.plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"{self.name}: confusion matrix",
        )

        self.visualizer.show_most_anomalous(
            raw["test_images"],
            raw["test_labels_original"],
            scores,
            top_k=16,
            title=f"{self.name}: most anomalous images",
        )

        self.visualizer.show_most_normal(
            raw["test_images"],
            raw["test_labels_original"],
            scores,
            top_k=16,
            title=f"{self.name}: most normal-looking images",
        )