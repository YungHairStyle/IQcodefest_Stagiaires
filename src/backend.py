from qiskit_ibm_runtime import QiskitRuntimeService

def get_ibm_backend():
    """
    Get the least busy IBM Quantum backend that is a simulator and operational.
    You must set up your IBM credentials using the following code:

    ```
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Only needed once
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token="YOUR_IBM_QUANTUM_API_TOKEN"
    )
    ```

    """

    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    print(f"Selected backend: {backend.name()}")

    return backend

