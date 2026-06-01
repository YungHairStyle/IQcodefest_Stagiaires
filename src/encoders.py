import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class BaseEncoder:
    """
    Base class for all encoders.

    Every encoder must implement:

        fit(X)
        transform(X)

    and will automatically inherit:

        fit_transform(X)

    The experiment class should only call this interface, without caring
    whether the encoder is PCA-based, amplitude-based, or something else.
    """

    def fit(self, X):
        """
        Learn any encoder parameters from training data.

        Examples:
            PCAAngleEncoder learns the scaler and PCA directions.
            AmplitudeEncoder does not really learn anything, but still
            implements this method for a consistent interface.
        """
        raise NotImplementedError

    def transform(self, X):
        """
        Transform raw data into encoded data.
        """
        raise NotImplementedError

    def fit_transform(self, X):
        """
        Convenience method: fit on X, then transform X.
        """
        self.fit(X)
        return self.transform(X)

    def get_info(self):
        """
        Return a dictionary with encoder metadata.
        """
        return {"type": "base_encoder"}


class PCAAngleEncoder(BaseEncoder):
    """
    PCA + angle-scaling encoder.

    This encoder implements the pipeline:

        raw 8x8 image
        -> flattened 64-dimensional vector
        -> normalization
        -> PCA to n_components
        -> scale each PCA coordinate to [0, pi]
        -> return angle vector

    If n_components = 4, then the quantum circuit usually uses 4 qubits.
    """

    def __init__(
        self,
        n_components=8,
        angle_range=(0.0, np.pi),
        standardize=True,
        random_state=7,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of PCA components to keep.
            This usually becomes the number of qubits for angle encoding.

        angle_range : tuple[float, float]
            Range to scale PCA features into.
            Usually (0, pi), because quantum rotation gates take angles.

        standardize : bool
            Whether to standardize pixels before PCA.
            Usually True.

        random_state : int
            Random seed for PCA reproducibility.
        """
        self.n_components = n_components
        self.angle_range = angle_range
        self.standardize = standardize
        self.random_state = random_state

        self.scaler = None
        self.pca = None
        self.angle_scaler = None

    def fit(self, X):
        """
        Fit the standardizer, PCA model, and angle scaler on training data.

        Important:
            In anomaly detection, this should be called only on normal
            training data. That way PCA learns the normal image manifold.

        Parameters
        ----------
        X : np.ndarray
            Shape: (n_samples, n_raw_features)
            Raw flattened image vectors.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)

        if self.standardize:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            X_scaled = X

        self.pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_pca = self.pca.fit_transform(X_scaled)

        self.angle_scaler = MinMaxScaler(feature_range=self.angle_range)
        self.angle_scaler.fit(X_pca)

        return self

    def transform(self, X):
        """
        Transform raw image vectors into PCA angle vectors.

        Parameters
        ----------
        X : np.ndarray
            Shape: (n_samples, n_raw_features)

        Returns
        -------
        X_angles : np.ndarray
            Shape: (n_samples, n_components)
            Each row contains angles to be used in a quantum feature map.
        """
        if self.pca is None or self.angle_scaler is None:
            raise RuntimeError("PCAAngleEncoder must be fitted before transform().")

        X = np.asarray(X, dtype=float)

        if self.standardize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        X_pca = self.pca.transform(X_scaled)
        X_angles = self.angle_scaler.transform(X_pca)

        return X_angles

    def get_info(self):
        """
        Return metadata about the PCA angle encoder.

        Useful for tables and presentations.
        """
        info = {
            "type": "pca_angle",
            "n_components": self.n_components,
            "angle_range": self.angle_range,
            "standardize": self.standardize,
        }

        if self.pca is not None:
            evr = self.pca.explained_variance_ratio_
            info["explained_variance_ratio"] = evr
            info["total_explained_variance"] = float(np.sum(evr))

        return info


class AmplitudeEncoder(BaseEncoder):
    """
    Amplitude encoder.

    This encoder implements:

        raw image vector x
        -> optionally pad to length 2^n
        -> normalize x so ||x|| = 1
        -> return x as a quantum statevector

    For sklearn digits:

        8x8 image = 64 pixels
        64 = 2^6

    So the whole image can be represented as amplitudes of a 6-qubit state.

    Conceptually:

        x = [x_0, x_1, ..., x_63]

    becomes:

        |psi_x> = sum_i x_i |i>

    after normalization.
    """

    def __init__(
        self,
        n_qubits=6,
        pad_to_power_of_two=True,
        eps=1e-12,
    ):
        """
        Parameters
        ----------
        pad_to_power_of_two : bool
            If True, pad input vectors with zeros up to length 2**n_qubits.

        eps : float
            Small number used to avoid division by zero.
        """
        self.n_qubits = n_qubits
        self.state_dim = 2**n_qubits
        self.normalize = True
        self.pad_to_power_of_two = pad_to_power_of_two
        self.eps = eps

        self.input_dim_ = None # Will be set during fit().

    def fit(self, X):
        """
        Store input dimension and validate compatibility.

        Parameters
        ----------
        X : np.ndarray
            Shape: (n_samples, n_raw_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        self.input_dim_ = X.shape[1]

        if self.input_dim_ > self.state_dim:
            raise ValueError(
                f"Input dimension {self.input_dim_} is larger than state dimension "
                f"2**n_qubits = {self.state_dim}. Increase n_qubits."
            )

        if self.input_dim_ != self.state_dim and not self.pad_to_power_of_two:
            raise ValueError(
                f"Input dimension {self.input_dim_} does not equal state dimension "
                f"{self.state_dim}, and pad_to_power_of_two=False."
            )

        return self

    def transform(self, X):
        """
        Convert raw image vectors into normalized amplitude vectors.

        Parameters
        ----------
        X : np.ndarray
            Shape: (n_samples, n_raw_features)

        Returns
        -------
        X_amp : np.ndarray
            Shape: (n_samples, 2**n_qubits)
            Each row has norm 1 and can be interpreted as a quantum statevector.
        """
        if self.input_dim_ is None:
            raise RuntimeError("AmplitudeEncoder must be fitted before transform().")

        X = np.asarray(X, dtype=float)

        if X.shape[1] != self.input_dim_:
            raise ValueError(
                f"Expected input dimension {self.input_dim_}, got {X.shape[1]}."
            )

        # Pad with zeros if the input vector is shorter than 2**n_qubits.
        if X.shape[1] < self.state_dim:
            pad_width = self.state_dim - X.shape[1]
            X_amp = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
        else:
            X_amp = X.copy()

        if self.normalize:
            norms = np.linalg.norm(X_amp, axis=1, keepdims=True)
            norms = np.maximum(norms, self.eps)
            X_amp = X_amp / norms

        return X_amp.astype(complex)

    def get_info(self):
        """
        Return metadata about the amplitude encoder.
        """
        return {
            "type": "amplitude",
            "n_qubits": self.n_qubits,
            "state_dim": self.state_dim,
            "input_dim": self.input_dim_,
            "normalize": self.normalize,
            "pad_to_power_of_two": self.pad_to_power_of_two,
        }