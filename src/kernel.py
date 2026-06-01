import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from qiskit_aer import AerSimulator
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator



class BaseKernel:
    """
    Base class for all kernels.
    """

    def fit_transform(self, X_train):
        """
        Compute the training kernel matrix.
        """
        raise NotImplementedError

    def transform(self, X_test, X_train=None):
        """
        Compute the test-vs-train kernel matrix.
        """
        raise NotImplementedError

    def get_info(self):
        """
        Return metadata dictionary.
        """
        return {"type": "base_kernel"}


class FidelityQuantumKernel(BaseKernel):
    """
    Quantum fidelity kernel.

    This class computes:

        K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2

    where phi is provided by a feature map object.

    The same kernel class works for:

        PCA angle encoding + AngleFeatureMap

    and:

        Amplitude encoding + AmplitudeFeatureMap

    because both feature maps expose:

        statevector(x)

    This is the key abstraction.
    """

    def __init__(self, feature_map, cache_train_states=True):
        """
        Parameters
        ----------
        feature_map : BaseFeatureMap
            Object with a statevector(x) method.

        cache_train_states : bool
            If True, store training statevectors after fit_transform().
            This makes transform(X_test) easier because it can reuse them.
        """
        self.feature_map = feature_map
        self.cache_train_states = cache_train_states

        self.X_train_ = None
        self.train_states_ = None

    def compute_statevectors(self, X):
        """
        Compute statevectors for a batch of encoded samples.

        Parameters
        ----------
        X : np.ndarray
            Encoded data.

        Returns
        -------
        states : np.ndarray
            Shape: (n_samples, state_dimension)
        """
        X = np.asarray(X)

        states = []
        for x in X:
            state = self.feature_map.statevector(x)
            states.append(state)

        return np.asarray(states, dtype=complex)

    def kernel_from_states(self, states_A, states_B):
        """
        Compute the fidelity kernel from two collections of statevectors.

        Parameters
        ----------
        states_A : np.ndarray
            Shape: (n_A, dim)

        states_B : np.ndarray
            Shape: (n_B, dim)

        Returns
        -------
        K : np.ndarray
            Shape: (n_A, n_B)
            K[i, j] = |<states_A[i] | states_B[j]>|^2
        """
        states_A = np.asarray(states_A, dtype=complex)
        states_B = np.asarray(states_B, dtype=complex)

        # Inner product matrix:
        # states_A @ states_B.conj().T gives <B|A> depending on convention,
        # but absolute value squared makes the convention irrelevant.
        K = np.abs(states_A @ states_B.conj().T) ** 2

        # Clean numerical noise.
        K = np.real(K)
        K = np.clip(K, 0.0, 1.0)

        return K


    def kernel_from_circuits(self, circuits_A, circuits_B, mode="aer", shots=5, backend=None):
        """
        Compute fidelity kernel using Qiskit AerSimulator from state-preparation circuits.

        Parameters
        ----------
        circuits_A : list[QuantumCircuit]
            Circuits preparing |psi_A>
        circuits_B : list[QuantumCircuit]
            Circuits preparing |psi_B>
        mode : str
            "statevector" -> exact fidelity from statevector simulation
            "aer" -> sampling-based estimate using AerSimulator
            "ibm" -> sampling-based estimate using real IBM hardware
        shots : int or None
            Number of shots for sampling-based modes. Ignored for "statevector".

        Returns
        -------
        K : np.ndarray
            Shape (n_A, n_B), K[i, j] = |<psi_A_i | psi_B_j>|^2
        """
        n_A = len(circuits_A)
        n_B = len(circuits_B)

        K = np.zeros((n_A, n_B), dtype=float)

        # Choose backend
        if mode == "statevector":
            shots = None
            backend = AerSimulator(method="statevector")
        elif mode == "aer":
            if shots is None:
                raise ValueError("Must specify shots for 'aer' mode.")
            backend = AerSimulator()
        elif mode == "ibm":
            if shots is None:
                raise ValueError("Must specify shots for 'ibm' mode.")
            if backend is None:
                raise ValueError("Must specify backend for 'ibm' mode.")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Transpile circuits
        circuits_A = [transpile(qc, backend) for qc in circuits_A]
        circuits_B = [transpile(qc, backend) for qc in circuits_B]

        for i, qc_a in enumerate(circuits_A):
            for j, qc_b in enumerate(circuits_B):

                # Build U_A^\dagger U_B acting on |0>
                qc = qc_b.copy()
                qc.compose(qc_a.inverse(), inplace=True)

                if mode == "statevector":
                    result = backend.run(qc).result()
                    final_state = result.get_statevector()

                    K[i, j] = np.abs(final_state[0]) ** 2

                elif mode in ["aer", "ibm"]:
                    qc.measure_all()

                    job = backend.run(qc, shots=shots)

                    result = job.result()

                    counts = result.get_counts()

                    zero_state = "0" * qc.num_qubits
                    K[i, j] = counts.get(zero_state, 0) / shots

        return np.clip(np.real(K), 0.0, 1.0)

    def fit_transform(self, X_train):
        """
        Compute and return the training kernel matrix.

        Parameters
        ----------
        X_train : np.ndarray
            Encoded training data.

        Returns
        -------
        K_train : np.ndarray
            Shape: (n_train, n_train)
        """
        self.X_train_ = np.asarray(X_train)
        train_states = self.compute_statevectors(self.X_train_)

        if self.cache_train_states:
            self.train_states_ = train_states

        K_train = self.kernel_from_states(train_states, train_states)

        return K_train

    def transform(self, X_test, X_train=None):
        """
        Compute the test-vs-train kernel matrix.

        Parameters
        ----------
        X_test : np.ndarray
            Encoded test data.

        X_train : np.ndarray or None
            Encoded training data.
            If None, use the cached training data/states from fit_transform().

        Returns
        -------
        K_test : np.ndarray
            Shape: (n_test, n_train)
        """
        X_test = np.asarray(X_test)

        test_states = self.compute_statevectors(X_test)

        if X_train is not None:
            train_states = self.compute_statevectors(X_train)
        else:
            if self.train_states_ is None:
                raise RuntimeError(
                    "No cached training states. Either call fit_transform() first "
                    "or pass X_train explicitly."
                )
            train_states = self.train_states_

        K_test = self.kernel_from_states(test_states, train_states)

        return K_test

    def sanity_check(self, K, symmetric=False):
        """
        Basic checks for a kernel matrix.

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix.

        symmetric : bool
            Set True for K_train. Set False for K_test.

        Returns
        -------
        checks : dict
            Dictionary of sanity check results.
        """
        K = np.asarray(K)

        checks = {
            "shape": K.shape,
            "min": float(np.min(K)),
            "max": float(np.max(K)),
            "has_nan": bool(np.isnan(K).any()),
            "in_range_0_1": bool(np.min(K) >= -1e-8 and np.max(K) <= 1.0 + 1e-8),
        }

        if symmetric:
            checks["symmetric"] = bool(np.allclose(K, K.T))
            checks["diagonal_close_to_one"] = bool(np.allclose(np.diag(K), 1.0))

        return checks

    def get_info(self):
        """
        Return metadata.
        """
        return {
            "type": "fidelity_quantum_kernel",
            "feature_map": self.feature_map.get_info(),
            "cache_train_states": self.cache_train_states,
        }


class ClassicalRBFKernel(BaseKernel):
    """
    Classical RBF kernel wrapper.

    This is useful because it lets us compare a classical kernel and a quantum
    kernel using the exact same OneClassSVMAnomalyModel.

    The RBF kernel is:

        K(x, y) = exp(-gamma ||x - y||^2)
    """

    def __init__(self, gamma="scale"):
        """
        Parameters
        ----------
        gamma : str or float
            If "scale", use 1 / (n_features * X.var()) during fit.
            If float, use that value directly.
        """
        self.gamma = gamma
        self.gamma_ = None
        self.X_train_ = None

    def _resolve_gamma(self, X):
        """
        Convert gamma='scale' to a numeric value.
        """
        if self.gamma == "scale":
            var = np.var(X)
            n_features = X.shape[1]
            if var == 0:
                return 1.0
            return 1.0 / (n_features * var)

        return float(self.gamma)

    def fit_transform(self, X_train):
        """
        Compute training RBF kernel.
        """
        X_train = np.asarray(X_train, dtype=float)

        self.X_train_ = X_train
        self.gamma_ = self._resolve_gamma(X_train)

        return rbf_kernel(X_train, X_train, gamma=self.gamma_)

    def transform(self, X_test, X_train=None):
        """
        Compute test-vs-train RBF kernel.
        """
        X_test = np.asarray(X_test, dtype=float)

        if X_train is None:
            if self.X_train_ is None:
                raise RuntimeError(
                    "No stored training data. Call fit_transform() first "
                    "or pass X_train explicitly."
                )
            X_train = self.X_train_
        else:
            X_train = np.asarray(X_train, dtype=float)

        if self.gamma_ is None:
            self.gamma_ = self._resolve_gamma(X_train)

        return rbf_kernel(X_test, X_train, gamma=self.gamma_)

    def get_info(self):
        """
        Return metadata.
        """
        return {
            "type": "classical_rbf_kernel",
            "gamma": self.gamma,
            "gamma_resolved": self.gamma_,
        }
