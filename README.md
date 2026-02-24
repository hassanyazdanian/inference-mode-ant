# Inference-Mode Ambient Noise Tomography

This repository contains the reference implementation accompanying the GJI submission:

**"Towards Inference-Mode Ambient Noise Tomography: A Framework for Phase Velocity Field Reconstruction and Uncertainty Quantification."**

It provides a fully reproducible synthetic experiment (regular station geometry) demonstrating Bayesian phase velocity reconstruction using:

- Whittle–Matérn Gaussian random field prior  
- Karhunen–Loève (KL) dimensionality reduction  
- Straight-ray line-integral forward operator  
- Maximum a posteriori (MAP) estimation (L-BFGS)  
- Full posterior sampling using NUTS  
- Posterior post-processing and uncertainty visualization  

The repository supports reproducible inference-mode imaging, where geological interpretation is grounded in posterior uncertainty quantification.

---

## 1. Required Packages

The recommended way to reproduce the results is via conda:

```bash
conda env create -f environment.yml
conda activate inference-mode-ant

## Main Dependencies

- **Python**: 3.12  
- **NumPy**: < 2.0 (for PyTorch compatibility)  
- **SciPy**  
- **Matplotlib**  
- **PyTorch**: 2.2  
- **Pyro**: 1.9.1  
- **scikit-image**  
- **ArviZ**

---

## Sanity Check

Run the following command to verify the environment:

```bash
python -c "import torch, pyro, numpy; print('Environment OK')"

## 2. Synthetic Field Generation

The synthetic velocity field is located in:


make_synthetic_field/


If needed, regenerate it via:

```bash
python make_synthetic_field/generate_synthetic_field.py

This produces:

synthetic_field.npy

True_field.png

The field defines two velocity levels representing geological structure.

## 3. Generate Synthetic Observations

Run:

```bash
python core/save_signal.py

This:

Loads the synthetic velocity field

Computes straight-ray travel times

Saves observation data to:

obs/obs_data.pickle