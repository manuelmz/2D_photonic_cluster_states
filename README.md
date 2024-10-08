# 2D_photonic_cluster_states

This repository contains the tools for the noisy quantum ssystem simulations in **Deterministic generation of a 20-qubit two-dimensional photonic cluster state** [arXiv:2409.06623](https://arxiv.org/abs/2409.06623v1). 
The simulations were implented using the open source library Cirq. The 2D-cluster state on a ladder graph is prepared on a register of $N$ qubits using two qutrits as the sources. 

To have a closer representation of the noise processes the source qutrits experience in the experiment, we constructed several qutrit functionalities, which include: amplitude and phase damping channels for qutrits, CPHASE and CNOT gates which include leakage to the third qutrit level, and coherent under-rotation errors.

To illustrate how these functionalities were used to obtain the simulation results in the paper we provide two short notebooks:

 - In the notebook fidelity_2D_states.ipynb we illustrate how to use these functionalitites to prepare a 2D cluster state and compute its fidelity.
 - In the notebook localizable_entanglement_2D_state.ipynb we illustrate how to use these functionalitites to compute the mean localizable entanglement of the two extremal qubits on the graph averaged over all the paths of length $N/2$ connecting them.

