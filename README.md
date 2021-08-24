# Exclusive Group Lasso for Structured Variable Selection

Libraries implementing the algorithms described in the paper "Exclusive Group Lasso for Structured Variable Selection", available [here](https://arxiv.org/abs/2108.10284).

The files are listed below with a short description. Check Matlab help (e.g., `help Subset`) for more information.
* **Subset:** Class that describes an exclusive group and implements the main functions acting on it.
* **proximal:** Proximal operator for the exclusive norm.
* **fista:** FISTA algorithm for minimization with the exclusive norm as regularizer.
* **activeset:** Implementation of the active set algorithm for minimization with the squared exclusive norm as regularizer.
* **activestring:** Implementation of the active set algorithm for minimization with the squared exclusive norm as regularizer. This is a modified version that looks for long strings of consecutive active entries.
* **solveRestrictedVar:** Algorithm to solve the restricted minimization problem with the squared exclusive norm as regularizer -- exploiting variational formulation.
* **solveRestrictedIP:** Algorithm to solve the restricted minimization problem with the squared exclusive norm as regularizer -- requires Matlab's Optimization Toolbox.
* **fistabasic:** Classic FISTA algorithm for minimization with the norm-1 regularizer.
* **proximalOverlap:** Proximal operator for the group lasso with overlap.
* **fistaOverlap:** FISTA algorithm for minimization with the "group lasso with overlap" regularizer.
* **test:** simple example script applying the above algorithms to the same support detection problem and comparing results.
