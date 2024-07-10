# 2D_photonic_cluster_states

In this repository we present the tools used in the support simulations for ----ref----. 
The simulations were done using the open source librayr Cirq. We construct the 2D cluster state on a register of $N$ qubits using two qutrits as the sources. 

In order to we constructed several qutrit functionalities, which include: amplitude and phase damping channels for qutrits, CPHASE and CNOT gate which include leakage to the third qutrit level, and coherent under rotation errors.

In the notebook fidelity_2D_states.ipynb we illustrate how to use these functionalitites to prepare a 2D cluster state and compute its fidelity.

In the notebook localizable_entanglement_2D_state.ipynb we illustrate how to use these functionalitites to compute the mean localizable entanglement of the two extremal qubits on the graph averaged over all the paths of length $N/2$ connecting them.
