Inference-Mode Ambient Noise Tomography

This repository contains the reference implementation accompanying the GJI submission:

Towards Inference-Mode Ambient Noise Tomography: A Framework for Phase Velocity Field Reconstruction and Uncertainty Quantification.

It provides a fully reproducible synthetic experiment (regular station geometry) demonstrating Bayesian phase velocity reconstruction using:

Whittle–Matérn Gaussian random field prior

Karhunen–Loève (KL) dimensionality reduction

Pushforward mapping for geological structure representation

Straight-ray line-integral forward operator

MAP estimation (L-BFGS)

Full posterior sampling using NUTS

Posterior post-processing and uncertainty visualization

The repository is designed to support reproducible inference-mode imaging, where geological interpretation is grounded in posterior uncertainty quantification.
