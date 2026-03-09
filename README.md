# velocity_power_spectrum
This repository provides a Python implementation of a model for the nonlinear velocity divergence power spectrum as a function of cosmological parameters.

The model predicts the nonlinear velocity divergence power spectrum as a function of:

- wavenumber $k$
- redshift $z$
- Hubble parameter $h$
- cold dark matter density parameter $\omega_{\rm cdm}$
  
The normalization convention is such that in linear theory the velocity divergence power spectrum coincides with the matter power spectrum.

# Description

The code implements a semi-empirical model calibrated on 'gevolution' simulations of the velocity field.

The model combines:
- a linear power spectrum computed with the cosmology code CLASS (http://class-code.net/)
- the evolution mapping approach to model the dependence on the Hubble factor $h$ (https://arxiv.org/pdf/2108.12710, https://arxiv.org/pdf/2406.08539)
- a nonlinear correction for modelling the dependence on $\omega_{\rm cdm}$, calibrated using N-body simulations
- a nonlinear damping calibrated in the fiducial cosmology

The model predicts
$P_{\theta\theta}(k, z)/[\mathcal{H}(z)f(z)]^2$, where $\mathcal{H}(z)$ and $f(z)$ are the conformal Hubble factor and the linear growth rate, respectively. This quantity reduces to the matter power spectrum in linear theory.

## Repository Structure

The repository is organized as follows:

```
.
├── velocity_model.py
├── compare_to_sim.ipynb
├── sim-data/
└── environment.yml
```

**velocity_model.py**  
Python file containing the model implementation.

**compare_to_sim.ipynb**  
Jupyter notebook that showcases how to compute the model for a given test cosmology and compare it to simulated data.

**sim-data/**  
Directory containing the simulation data used to estimate and validate the model.
These spectra are computed using the N-body code 'gevolution' (https://github.com/gevolution-code/gevolution-1.3)

**environment.yml**  
Conda environment file specifying the Python packages and versions required to run the notebook.

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/username/velocity-divergence-model.git
cd velocity-divergence-model
```

2. **Create the conda environment** from the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

2. **Activate the environment** :

```bash
conda activate env-velocity
```

3. **Add the environment as a Jupyter kernel** :

```bash
python -m ipykernel install --user --name kernel-velocity --display-name "kernel-velocity"
```
4. **Launch Jupyter Notebook or JupyterLab** and select the kernel kernel-velocity.

## Acknowledgements

The code provided in this repository uses the CLASS cosmology code (http://class-code.net/).
The simulated power spectra are computed using the N-body code 'gevolution' (https://github.com/gevolution-code/gevolution-1.3).
