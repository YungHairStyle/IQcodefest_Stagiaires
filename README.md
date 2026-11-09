# Quantum-kernel anomaly detection — IQ Codefest

Team challenge project exploring quantum-kernel methods for anomaly detection alongside classical comparisons. The repository contains notebook experiments with dimensionality reduction and amplitude-based encoding, together with reusable experiment components.

## Where to start

- `quantum_similarity.ipynb` and `test_1.ipynb`: notebook experiments.
- `src/experiment.py`: coordinates data loading, encoding, kernel construction, model fitting, and evaluation.
- `src/dataset.py`, `encoders.py`, `feature_map.py`, and `kernel.py`: data and feature representations.
- `src/models.py`, `anomaly_eval.py`, and `visualizer.py`: scoring, metrics, and plots.
- `amp_classical/`, `amplitude_quantum/`, `pca_classical/`, and `PCA_quantum/`: experiment variants and outputs.
- `challenge/`: challenge materials.

## Reproducing an experiment

Use an isolated Python environment and inspect `requirements.txt` and the notebook imports before installation. The requirements file records a broad development environment. Open the notebooks from the repository root and review their dataset paths and experiment settings before executing them.

The code supports comparisons of encodings and kernels; the presence of a quantum kernel does not establish a quantum advantage. Results depend on dataset selection, preprocessing, hyperparameters, and simulation resources.

This is collaborative Codefest work. Repository ownership should not be interpreted as sole authorship of the team project.

---
[Asaad Hanna — research, CV, and contact](https://yunghairstyle.github.io/)

