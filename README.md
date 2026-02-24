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
```
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
```

## 2. Synthetic Field Generation

The synthetic velocity field is located in:


make_synthetic_field/


If needed, regenerate it via:

```bash
python make_synthetic_field/generate_synthetic_field.py
```

This produces:

synthetic_field.npy

True_field.png

The field defines two velocity levels representing geological structure.

## 3. Generate Synthetic Observations

Run:

```bash
python core/save_signal.py
```

This:

Loads the synthetic velocity field

Computes straight-ray travel times

Saves observation data to:

```bash
obs/obs_data.pickle
```

## 4. MAP Estimation (L-BFGS)

Run:

```bash
python core/MAP_estimate.py --device cpu
```

This computes the maximum a posteriori (MAP) estimate of the phase velocity field under:

- Whittle–Matérn prior

- KL parameterization

- Straight-ray forward operator


## 5. Posterior Sampling (NUTS)

Run:

```bash
python core/sampling.py --device cpu
```

This performs full Bayesian posterior sampling using:

- Hamiltonian Monte Carlo (NUTS)

- Pyro backend

- Reduced KL parameterization

Samples are saved in:

```bash
stat/
```


## 6. Posterior Post-Processing

Run:

```bash
python core/post_process.py
```

This generates:

- Posterior mean field

- Posterior standard deviation (UQ)

- Sampling diagnostics (ESS, trace plots)


## Core Components

### `core/line_integral.py`

Implements the straight-ray forward operator used to compute travel times along station pairs.

### `core/prior.py`

Defines the Whittle–Matérn Gaussian prior and KL expansion used to parameterize the velocity field.

These two modules form the mathematical backbone of the inference engine.

---

## Important

All commands must be executed from the repository root:

```bash
inference-mode-ant/
```

## Full Reproduction Pipeline

```bash
conda env create -f environment.yml
conda activate inference-mode-ant

python make_synthetic_field/generate_synthetic_field.py
python core/save_signal.py
python core/MAP_estimate.py --device cpu
python core/sampling.py --device cpu
python core/post_process.py
```

## Output

The full pipeline produces:

- True synthetic field  
- MAP reconstruction  
- Posterior mean  
- Posterior uncertainty maps  
- Sampling diagnostics  
