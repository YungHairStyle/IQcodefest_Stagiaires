import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class BaseFeatureMap:
    """
    Base class for all quantum feature maps.

    Every feature map should be able to produce:

        1. A circuit, for visualization or hardware-style execution.
        2. A statevector, for fast exact kernel computation in simulation.
    """

    def build_circuit(self, x):
        """
        Build a Qiskit circuit that encodes one data point.
        """
        raise NotImplementedError

    def statevector(self, x):
        """
        Return the quantum statevector corresponding to one encoded data point.
        """
        raise NotImplementedError

    def get_info(self):
        """
        Return a metadata dictionary.
        """
        return {"type": "base_feature_map"}


class AngleFeatureMap(BaseFeatureMap):
    """
    Angle-based quantum feature map.

    This feature map expects an input vector of angles:

        x = [x_0, x_1, ..., x_{n-1}]

    and builds an n-qubit quantum circuit.

    The circuit structure is:

        1. Optionally apply H to all qubits.
        2. Apply Ry(x_i) and Rz(x_i) to each qubit.
        3. Apply entangling blocks that depend on products x_i x_j.
        4. Repeat for reps layers.

    This creates a nonlinear quantum embedding because the entangling angles
    contain pairwise feature products.

    The quantum kernel will later compare states produced by this feature map.
    """

    def __init__(
        self,
        reps=2,
        entanglement="linear",
        include_hadamards=True,
        include_ring=True,
    ):
        """
        Parameters
        ----------
        reps : int
            Number of repeated encoding/entangling layers.

        entanglement : str
            Entanglement pattern. Currently supports:
                "linear"
            You can extend this later.

        include_hadamards : bool
            Whether to start with H gates on all qubits.

        include_ring : bool
            Whether to also entangle the last qubit with the first qubit
            when there are more than 2 qubits.
        """
        self.reps = reps
        self.entanglement = entanglement
        self.include_hadamards = include_hadamards
        self.include_ring = include_ring

        if self.entanglement not in ["linear"]:
            raise ValueError("Currently only entanglement='linear' is supported.")

    def build_circuit(self, x):
        """
        Build the angle feature map circuit for a single encoded sample.

        Parameters
        ----------
        x : np.ndarray
            Shape: (n_features,)
            The entries should be angles, typically in [0, pi].

        Returns
        -------
        qc : QuantumCircuit
            Quantum circuit encoding x.
        """
        x = np.asarray(x, dtype=float)
        n_qubits = len(x)

        qc = QuantumCircuit(n_qubits)

        if self.include_hadamards:
            for q in range(n_qubits):
                qc.h(q)

        for _ in range(self.reps):
            # Local angle encoding.
            for q in range(n_qubits):
                qc.ry(float(x[q]), q)
                qc.rz(float(x[q]), q)

            # Nearest-neighbor entangling feature interactions.
            for q in range(n_qubits - 1):
                angle = float((x[q] * x[q + 1]) / np.pi)
                qc.cx(q, q + 1)
                qc.rz(angle, q + 1)
                qc.cx(q, q + 1)

            # Optional ring connection between last and first qubit.
            if self.include_ring and n_qubits > 2:
                angle = float((x[-1] * x[0]) / np.pi)
                qc.cx(n_qubits - 1, 0)
                qc.rz(angle, 0)
                qc.cx(n_qubits - 1, 0)

        return qc

    def statevector(self, x):
        """
        Compute the exact statevector produced by the angle feature map.

        Parameters
        ----------
        x : np.ndarray
            Shape: (n_features,)

        Returns
        -------
        state : np.ndarray
            Complex statevector of shape (2**n_features,)
        """
        qc = self.build_circuit(x)
        return Statevector.from_instruction(qc).data

    def get_info(self):
        """
        Return metadata about the feature map.
        """
        return {
            "type": "angle_feature_map",
            "reps": self.reps,
            "entanglement": self.entanglement,
            "include_hadamards": self.include_hadamards,
            "include_ring": self.include_ring,
        }


class AmplitudeFeatureMap(BaseFeatureMap):
    """
    Amplitude feature map.

    This feature map expects the input x to already be a normalized amplitude
    vector.

    For the sklearn digits dataset:

        x.shape = (64,)

    Since 64 = 2^6, this corresponds to a 6-qubit state.

    In exact statevector simulation, the fastest method is simply:

        return x

    because the AmplitudeEncoder already created the normalized quantum state.

    For visualization, build_circuit(x) constructs a Qiskit circuit using
    initialize(x, qubits).
    """

    def __init__(self, validate_norm=True, atol=1e-8):
        """
        Parameters
        ----------
        validate_norm : bool
            Whether to check that input vectors have norm 1.

        atol : float
            Tolerance for norm checking.
        """
        self.validate_norm = validate_norm
        self.atol = atol

    def _num_qubits_from_state(self, x):
        """
        Infer number of qubits from the length of an amplitude vector.
        """
        dim = len(x)
        n_qubits_float = np.log2(dim)
        n_qubits = int(round(n_qubits_float))

        if 2**n_qubits != dim:
            raise ValueError(
                f"Amplitude vector length {dim} is not a power of two."
            )

        return n_qubits

    def _validate(self, x):
        """
        Validate an amplitude vector.
        """
        x = np.asarray(x, dtype=complex)

        self._num_qubits_from_state(x)

        if self.validate_norm:
            norm = np.linalg.norm(x)
            if not np.isclose(norm, 1.0, atol=self.atol):
                raise ValueError(
                    f"Amplitude vector must have norm 1. Got norm={norm}."
                )

        return x

    def build_circuit(self, x):
        """
        Build a Qiskit circuit that amplitude-encodes x.

        Parameters
        ----------
        x : np.ndarray
            Shape: (2**n_qubits,)
            Normalized amplitude vector.

        Returns
        -------
        qc : QuantumCircuit
            Circuit that initializes |x>.
        """
        x = self._validate(x)
        n_qubits = self._num_qubits_from_state(x)

        qc = QuantumCircuit(n_qubits)
        qc.initialize(x, list(range(n_qubits)))

        return qc

    def statevector(self, x):
        """
        Return x directly as the statevector.

        This is much faster than asking Qiskit to simulate initialize() every
        time, and it is mathematically equivalent for the ideal statevector
        kernel.
        """
        x = self._validate(x)
        return x

    def get_info(self):
        """
        Return metadata about the feature map.
        """
        return {
            "type": "amplitude_feature_map",
            "validate_norm": self.validate_norm,
            "atol": self.atol,
        }