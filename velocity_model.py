"""
This module contains the implementation of the non-linear model for the velocity power spectrum, which uses an evolution mapping
approach (https://arxiv.org/pdf/2108.12710, https://arxiv.org/pdf/2406.08539).
An example of how to use the model and compare it to simulations is provided in the Jupyter notebook "compare_to_sim.ipynb".

Author: Francesca Lepori
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.optimize import brentq

from classy import Class

# =====================================================================
# FiducialCosmology (FIDUCIAL)
# =====================================================================

@dataclass(frozen=True)
class FiducialCosmology:
    """Fiducial cosmological parameters, shared across the whole module.

    Flat LCDM, massless neutrinos (N_ur only, no massive species). `h` is
    the fiducial Hubble parameter that most functions below default to
    unless a "test" cosmology is passed explicitly.
    """
    h: float = 0.67556
    omega_b: float = 0.022032
    omega_cdm: float = 0.12038
    A_s: float = 2.215e-9
    n_s: float = 0.9619
    k_pivot: float = 0.05        # 1/Mpc (NOT h/Mpc!)
    T_cmb: float = 2.7255        # K
    N_ur: float = 3.046

    def to_class_params(self):
        """Base CLASS parameter dict; caller still needs to set 'output'/'z_pk'/etc."""
        return {
            'h': self.h, 'omega_b': self.omega_b, 'omega_cdm': self.omega_cdm,
            'k_pivot': self.k_pivot, 'A_s': self.A_s, 'n_s': self.n_s,
            'T_cmb': self.T_cmb, 'N_ur': self.N_ur,
        }


FIDUCIAL = FiducialCosmology()

# --- backward-compatible flat aliases ---
h         = FIDUCIAL.h
h_fid     = FIDUCIAL.h
omega_b   = FIDUCIAL.omega_b
omega_cdm = FIDUCIAL.omega_cdm
A_s       = FIDUCIAL.A_s
n_s       = FIDUCIAL.n_s
k_pivot   = FIDUCIAL.k_pivot
T_cmb     = FIDUCIAL.T_cmb
N_ur      = FIDUCIAL.N_ur


# =====================================================================
# Fiducial simulation -> redshift mapping 
# =====================================================================

redshifts = {
    "pk000": 2.0, "pk001": 1.9, "pk002": 1.8, "pk003": 1.7, "pk004": 1.6,
    "pk005": 1.5, "pk006": 1.4, "pk007": 1.3, "pk008": 1.2, "pk009": 1.1,
    "pk010": 1.0, "pk011": 0.9, "pk012": 0.8, "pk013": 0.7, "pk014": 0.6,
    "pk015": 0.5, "pk016": 0.4, "pk017": 0.3, "pk018": 0.2, "pk019": 0.1,
    "pk020": 0.0,
}
zfid_arr = np.array(sorted(redshifts.values(), reverse=True))


# Function to read gevolution power spectra
def read_gevolution_outputs(path, h, redshifts=redshifts,
                            filename_template="lcdm_box128_{tag}_{field}.dat"):
    """Read gevolution theta/omega(vorticity)/delta power spectra.

    Parameters
    ----------
    path : str
        Directory containing the gevolution output files.
    h : float
        Hubble parameter of *this* simulation, used only for unit conversion
        (Pk / h^2, or / h^3 for delta -- matches the gevolution output
        convention, not a cosmology choice).
    redshifts : dict
        {'pkNNN': z} snapshot-tag -> redshift mapping.
    filename_template : str
        Format string with `{tag}` (e.g. 'pk000') and `{field}`
        ('theta', 'omega', 'delta') placeholders. Defaults to the
        'lcdm_box128_pkNNN_theta.dat' convention used by the fiducial
        h-derivative runs; override per run, e.g. "lcdm_{tag}_{field}.dat"
        for runs that don't tag the box size in the filename.

    Returns
    -------
    dict
        {z: {"k_theta", "Pk_theta", "Nk_theta", "k_vort", "Pk_vort",
             "Nk_vort", "k_delta", "Pk_delta", "Nk_delta"}}
    """
    results = {}
    for tag, z in redshifts.items():
        # theta
        data_theta = np.loadtxt(os.path.join(path, filename_template.format(tag=tag, field="theta")))
        k_theta = data_theta[:, 0]
        Pk_theta = data_theta[:, 1] * 2.0 * np.pi**2 / (k_theta**3) / h / h
        Nk_theta = data_theta[:, 4]  # number of modes for theta

        # vorticity
        data_vort = np.loadtxt(os.path.join(path, filename_template.format(tag=tag, field="omega")))
        k_vort = data_vort[:, 0]
        Pk_vort = data_vort[:, 1] * 2.0 * np.pi**2 / (k_vort**3) / h / h
        Nk_vort = data_vort[:, 4]  # number of modes for vorticity

        # density
        data_dens = np.loadtxt(os.path.join(path, filename_template.format(tag=tag, field="delta")))
        k_delta = data_dens[:, 0]
        Pk_delta = data_dens[:, 1] * 2.0 * np.pi**2 / (k_delta**3)
        Nk_delta = data_dens[:, 4]  # number of modes for density

        results[z] = {
            "k_theta": k_theta,
            "Pk_theta": Pk_theta,
            "Nk_theta": Nk_theta,
            "k_vort": k_vort,
            "Pk_vort": Pk_vort,
            "Nk_vort": Nk_vort,
            "k_delta": k_delta,
            "Pk_delta": Pk_delta,
            "Nk_delta": Nk_delta,
        }
    return results


# =====================================================================
# CLASS background helpers
# =====================================================================

@lru_cache(maxsize=512)
def compute_class_background(h, omega_b, omega_cdm, k_pivot, n_s, T_cmb, N_ur, As_fid, z):
    params = {
        'h': h,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'k_pivot': k_pivot,
        'A_s': As_fid,
        'n_s': n_s,
        'T_cmb': T_cmb,
        'N_ur': N_ur,
    }

    cosmo = Class()
    cosmo.set(params)
    cosmo.set({
        'output': 'mPk, dTk, vTk',
        'lensing': 'no',
        'z_pk': '0.0, 0.1, 1.0, 2.0, 3.0',
        'P_k_max_h/Mpc': 300.0,
    })
    cosmo.compute()

    Hconf = cosmo.Hubble(z)/(1.0+z)/h
    f_growth = cosmo.scale_independent_growth_factor_f(z)

    cosmo.struct_cleanup()
    cosmo.empty()
    return Hconf, f_growth


def compute_class_background_from_h_As(h_val, A_s, z=0.0):
    return compute_class_background(
        h_val, omega_b, omega_cdm, k_pivot, n_s, T_cmb, N_ur,
        A_s, z=z
    )

# Compute the derivative of power spectra with respect to h using a 4th-order central finite difference formula, for each redshift and spectrum.
# We estimate derivatives at fixed sigma_12, so we need to build a lookup table that maps sigma_12 --> redshift for each h value,
# using the pre-computed Pk_redshifts array.
# A_s values are fixed to the fiducial value.

# =====================================================================
# Finite-difference-in-h derivative setup (single definition)
# =====================================================================
#
# base_fiducial / base_h_folder / h_folders / h_values must point at gevolution
# output directories that exist on disk; sigma12_fid / Pk_redshifts are
# precomputed elsewhere (see get_z_fixed_sigma12 notebook) and must stay
# index-aligned with h_values / h_folders. There is no automatic check that
# a given h_folders[i] corresponds to h_values[i] and Pk_redshifts[i] --
# double-check this by hand if you add or reorder entries.

# --- User input ---
base_fiducial = 'sim-data/Ngrid1024Lbox256/h-deriv/h_0.67556_As_2.215e-9/'
base_h_folder = 'sim-data/Ngrid1024Lbox256/h-deriv/'

dh = 0.025 * h_fid  # step size for finite difference

h_folders = [
    "h_0.64178_As_2.215e-9",
    "h_0.65867_As_2.215e-9",
    "h_0.69245_As_2.215e-9",
    "h_0.70934_As_2.215e-9",
]
h_values = [0.64178, 0.65867, 0.69245, 0.70934]
As_values = np.array([A_s] * len(h_values))  # A_s values fixed to the fiducial value

# Module-level cache for derivative interpolator (lazy-built)
_derivatives_ready = False
_deriv_theta_interp = None
_derivatives_h = None
_sigma_grid = None
_logk_grid = None

# Precomputed fiducial sigma12 values at zfid_arr
sigma12_fid = np.array([
    0.35155175, 0.36312531, 0.37544780, 0.38858837, 0.40262385,
    0.41763932, 0.43372919, 0.45099765, 0.46955930, 0.48953958,
    0.51107410, 0.53430825, 0.55939401, 0.58648681, 0.61573799,
    0.64728486, 0.68123585, 0.71764858, 0.75650423, 0.79767330,
    0.84087914
])

# Pk_redshifts array: strings of redshifts for each h value (index-aligned
# with h_values / h_folders above)
Pk_redshifts = [
    "2.0062, 1.9066, 1.8071, 1.7076, 1.6082, 1.5088, 1.4095, 1.3103, 1.2112, 1.1122, 1.0134, 0.9147, 0.8163, 0.7180, 0.6200, 0.5224, 0.4251, 0.3284, 0.2322, 0.1367, 0.0419",
    "2.0031, 1.9034, 1.8036, 1.7038, 1.6041, 1.5045, 1.4048, 1.3052, 1.2057, 1.1062, 1.0068, 0.9075, 0.8082, 0.7091, 0.6102, 0.5114, 0.4128, 0.3144, 0.2164, 0.1187, 0.0215",
    "1.9968, 1.8966, 1.7963, 1.6960, 1.5958, 1.4954, 1.3950, 1.2946, 1.1942, 1.0936, 0.9930, 0.8923, 0.7915, 0.6906, 0.5895, 0.4883, 0.3868, 0.2851, 0.1830, 0.0805, -0.0224",
    "1.9935, 1.8930, 1.7925, 1.6920, 1.5914, 1.4907, 1.3900, 1.2891, 1.1882, 1.0871, 0.9858, 0.8844, 0.7828, 0.6809, 0.5787, 0.4762, 0.3732, 0.2697, 0.1654, 0.0603, -0.0458",
]

"""
def discover_h_derivative_folders(base_h_folder=base_h_folder, pattern=r"h_([0-9.]+)_As_"):
    Optional: auto-discover (h_value, folder_name) pairs from disk instead
    of maintaining `h_folders` / `h_values` by hand.

    Not used by default (compute_derivatives_h still defaults to the
    hand-maintained `h_folders`/`h_values` above, since `Pk_redshifts` must
    stay index-aligned with them and that alignment isn't derivable from
    the folder names alone). Use this to sanity-check that the hand-typed
    lists actually match what's on disk, or as a starting point if you add
    new h-derivative runs:

        >>> pairs = discover_h_derivative_folders()
        >>> [h for h, _ in pairs] == h_values  # sanity check
    
    import re
    if not os.path.isdir(base_h_folder):
        raise FileNotFoundError(f"base_h_folder does not exist: {base_h_folder}")
    found = []
    for name in sorted(os.listdir(base_h_folder)):
        m = re.match(pattern, name)
        if m and os.path.isdir(os.path.join(base_h_folder, name)):
            found.append((float(m.group(1)), name))
    return sorted(found, key=lambda pair: pair[0])
"""

def compute_derivatives_h(base_fiducial=base_fiducial,
                          base_h_folder=base_h_folder,
                          h_fid=h_fid,
                          h_values=h_values,
                          As_values=As_values,
                          h_folders=h_folders,
                          dh=dh,
                          zfid_arr=zfid_arr,
                          sigma12_fid=sigma12_fid,
                          Pk_redshifts=Pk_redshifts):
    """
    Compute derivatives dP/dh from simulation outputs and CLASS backgrounds.
    Returns the `derivatives_h` dictionary.
    """
    # Convert Pk_redshifts strings → numeric arrays
    z_arrays = [np.array([float(x) for x in row.split(',')]) for row in Pk_redshifts]

    # Sanity check
    assert len(h_values) == len(z_arrays)
    assert len(sigma12_fid) == len(z_arrays[0])

    # Build lookup table
    redshift_data = {}
    for h, z_vals in zip(h_values, z_arrays):
        redshift_data[h] = {
            float(f"{sigma:.6f}"): float(f"{z:.6f}")
            for sigma, z in zip(sigma12_fid, z_vals)
        }

    # --- Precompute CLASS background at low z for spline extrapolation ---
    z_grid = np.linspace(0.0, 0.2, 11)
    H_splines = {}
    f_splines = {}
    for h_val, As_val in zip(h_values, As_values):
        H_vals, f_vals = [], []
        for z0 in z_grid:
            Hz, fz = compute_class_background_from_h_As(h_val, As_val, z=z0)
            H_vals.append(Hz)
            f_vals.append(fz)
        H_splines[h_val] = CubicSpline(z_grid, H_vals, extrapolate=True)
        f_splines[h_val] = CubicSpline(z_grid, f_vals, extrapolate=True)

    # --- Read all data ---
    all_der_h = {}
    all_der_h[f"fiducial={h_fid:.5f}"] = read_gevolution_outputs(base_fiducial, h_fid)
    for val, folder in zip(h_values, h_folders):
        path_var = os.path.join(base_h_folder, folder)
        all_der_h[f"h={val:.5f}"] = read_gevolution_outputs(path_var, val)

    # --- Compute dP/dh for each redshift and spectrum ---
    derivatives_h = {}
    # Loop through index_z = 20, 19, ..., 0
    for index_z in range(len(zfid_arr)-1, -1, -1):
        z_fid = zfid_arr[index_z]

        P_spectra_theta = []
        P_spectra_vort = []
        P_spectra_vel = []
        P_spectra_delta = []

        for h_val, As_val in zip(h_values, As_values):
            # Deterministic lookup: map sigma (from fiducial grid) -> redshift
            sigma = sigma12_fid[index_z]
            key = float(f"{sigma:.6f}")
            # Prefer exact formatted key; if missing, pick nearest available sigma key
            if key in redshift_data[h_val]:
                z = redshift_data[h_val][key]
            else:
                # fallback: choose nearest sigma key to avoid ordering issues
                available = np.array(list(redshift_data[h_val].keys()))
                idx_nearest = np.argmin(np.abs(available - sigma))
                nearest_key = float(f"{available[idx_nearest]:.6f}")
                z = redshift_data[h_val][nearest_key]
            results = all_der_h[f"h={h_val:.5f}"]

            if z >= 0:
                Hconf, f_growth = compute_class_background_from_h_As(h_val, As_val, z=z)
            else:
                Hconf = H_splines[h_val](z)
                f_growth = f_splines[h_val](z)

            k = results[z_fid]["k_vort"] * h_val
            Pk_vort = results[z_fid]["Pk_vort"]
            Pk_theta = results[z_fid]["Pk_theta"]
            Pk_delta = results[z_fid]["Pk_delta"]
            spec_theta = (Pk_theta) / (Hconf * f_growth)**2 / h_val**3
            spec_vort = (Pk_vort) / (Hconf * f_growth)**2 / h_val**3
            spec_vel = (Pk_theta + Pk_vort) / (Hconf * f_growth)**2 / h_val**3
            spec_delta = Pk_delta / h_val**3

            P_spectra_theta.append(spec_theta)
            P_spectra_vort.append(spec_vort)
            P_spectra_vel.append(spec_vel)
            P_spectra_delta.append(spec_delta)

        dPtheta_dh = (-P_spectra_theta[3] + 8.0*P_spectra_theta[2] - 8.0*P_spectra_theta[1] + P_spectra_theta[0]) / (12.0*dh)
        dPvort_dh = (-P_spectra_vort[3] + 8.0*P_spectra_vort[2] - 8.0*P_spectra_vort[1] + P_spectra_vort[0]) / (12.0*dh)
        dPvel_dh = (-P_spectra_vel[3] + 8.0*P_spectra_vel[2] - 8.0*P_spectra_vel[1] + P_spectra_vel[0]) / (12.0*dh)
        dPdelta_dh = (-P_spectra_delta[3] + 8.0*P_spectra_delta[2] - 8.0*P_spectra_delta[1] + P_spectra_delta[0]) / (12.0*dh)

        derivatives_h[f"z_index_{index_z}"] = {
            "k": k,
            "dPtheta_dh": dPtheta_dh,
            "dPvort_dh": dPvort_dh,
            "dPvel_dh": dPvel_dh,
            "dPdelta_dh": dPdelta_dh,
        }

    return derivatives_h


def build_deriv_interpolator(derivatives_h):
    """
    Build a RegularGridInterpolator for dP_theta/dh from derivatives_h dict.
    Returns (deriv_theta_interp, sigma_grid, logk_grid).
    """
    # Sort indices to ensure consistent order
    indices = sorted([int(key.split("_")[-1]) for key in derivatives_h.keys()])

    # sigma_12 values corresponding to the indices
    sigma_grid = np.array([sigma12_fid[i] for i in indices])

    # k grid (assume identical for all sigma)
    k_grid = derivatives_h[f"z_index_{indices[0]}"]["k"]

    # Build 2D array of dPtheta/dh
    Dtheta_grid = np.array([
        derivatives_h[f"z_index_{i}"]["dPtheta_dh"]
        for i in indices
    ])  # shape = (N_sigma, N_k)

    logk_grid = np.log(k_grid)

    deriv_theta_interp = RegularGridInterpolator(
        (sigma_grid, logk_grid),
        Dtheta_grid,
        bounds_error=False,
        fill_value=None  # allow extrapolation along sigma
    )

    return deriv_theta_interp, sigma_grid, logk_grid


def prepare_derivatives():
    """Prepare derivative interpolator and cache it in module-level variables."""
    global _derivatives_ready, _deriv_theta_interp, _derivatives_h, _sigma_grid, _logk_grid
    if _derivatives_ready:
        return

    # Compute derivatives and build interpolator (this is the expensive part)
    _derivatives_h = compute_derivatives_h()
    _deriv_theta_interp, _sigma_grid, _logk_grid = build_deriv_interpolator(_derivatives_h)
    _derivatives_ready = True

def dPtheta_dh_at_sigma(k, sigma_fixed):
    """Compute dP_theta/dh at fixed sigma for multiple k values.
    This function will prepare and cache the derivative interpolator
    on first use (so the expensive computation is not done at import).
    """
    prepare_derivatives()

    k = np.atleast_1d(k)
    sigma_array = np.full_like(k, sigma_fixed)
    logk = np.log(k)

    points = np.column_stack([sigma_array, logk])
    values = _deriv_theta_interp(points)

    # Mask points where logk is outside the grid and set to zero
    logk_min, logk_max = _logk_grid[0], _logk_grid[-1]
    outside_mask = (logk < logk_min) | (logk > logk_max)
    values[outside_mask] = 0.0

    return values

# -----------------------------
# Window functions
# -----------------------------

def W_tophat(x):
    """Fourier transform of a real-space top-hat window."""
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3

# -----------------------------
# Sigma_R
# -----------------------------

def sigma_R(cosmo, R_mpc, z=0.0):
    """
    Compute sigma_R using CLASS linear power spectrum.
    """
    def integrand(k):
        P = cosmo.pk_lin(k, z)
        W = W_tophat(k * R_mpc)
        return k**2 * P * W**2

    integral = integrate.quad(integrand, 1e-4, 100, limit=200)[0]
    return np.sqrt(integral / (2 * np.pi**2))

def compute_sigma12(A_s_x, h_x, omega_cdm_x, R=12.0, z=0.0, give_growth=False,
                    growth_zmin=0.0, growth_zmax=3.0, growth_nz=201):
    """Compute sigma_12 for given A_s, h and omega_cdm.

    If `give_growth` is False (default) the function returns a scalar sigma12 at
    redshift `z` (computed with CLASS). If `give_growth` is True the function
    returns a tuple `(sigma12, D_spline)` where `D_spline` is a CubicSpline that
    evaluates the linear growth factor D(z)/D(0) at arbitrary redshift.

    Parameters
    ----------
    growth_zmin, growth_zmax, growth_nz : float,int
        Grid used to compute the growth spline when `give_growth` is True.
    """
    # If growth interpolator requested, ask CLASS to compute background on a z-grid
    if give_growth:
        z_grid = np.linspace(growth_zmin, growth_zmax, growth_nz)
        #z_pk_str = ','.join([f"{zz:.6f}" for zz in z_grid])
        params = {
            'h': h_x,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm_x,
            'k_pivot': k_pivot,
            'A_s': A_s_x,
            'n_s': n_s,
            'T_cmb': T_cmb,
            'N_ur': N_ur,
            'output': 'mPk',
            'z_pk': f"{growth_zmin},{z},{growth_zmax}",
            'P_k_max_h/Mpc': 300.0,
            'non_linear': 'no'
        }
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        # sigma at requested z
        sigma12 = sigma_R(cosmo, R, z)

        # build growth factor grid and spline (normalize to D(0)=1)
        D_vals = np.array([cosmo.scale_independent_growth_factor(zz) for zz in z_grid])
        # normalize so D(0)=1 (find index closest to z=0)
        idx0 = np.argmin(np.abs(z_grid - 0.0))
        D_vals = D_vals / float(D_vals[idx0])

        from scipy.interpolate import CubicSpline as _CubicSpline
        D_spline = _CubicSpline(z_grid, D_vals, extrapolate=True)

        cosmo.struct_cleanup()
        cosmo.empty()

        return sigma12, D_spline

    # Default: only compute sigma12 at single z
    params = {
            'h': h_x,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm_x,
            'k_pivot': k_pivot,
            'A_s': A_s_x,
            'n_s': n_s,
            'T_cmb': T_cmb,
            'N_ur': N_ur,
            'output': 'mPk',
            'z_pk': str(z),
            'P_k_max_h/Mpc': 300.0,
            'non_linear': 'no'
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    sigma12 = sigma_R(cosmo, R, z)
    cosmo.struct_cleanup()
    cosmo.empty()
    return sigma12

# ----------------------------------------------------------
# Model ratio of nonlinear spectra with different omega_cdm
# ----------------------------------------------------------

def W_k_sigma12(k, sigma12, k0, alpha):
    k_NL = k0 * sigma12**(-alpha)
    return 2.0 / (1.0 + np.exp(k / k_NL))

def model_ratio_nl(k, ratio_lin, sigma12, k0, alpha):
    W = W_k_sigma12(k, sigma12, k0, alpha)
    return 1.0 + W * (ratio_lin - 1.0)


# ----------------------------------------------------------------
# Model ration of nonlinear to linear spectra (fiducial cosmology)
# ----------------------------------------------------------------

def kstar_stable(sigma12, k_high, beta, k_low, alpha):
    """
    Smooth hybrid function for k_star(sigma12):
    - Exponential dominates at small sigma12 (high-z)
    - Power-law dominates at large sigma12 (low-z)
    """
    return k_high * np.exp(-beta * sigma12) + k_low * sigma12**(-alpha)


def B_hyperbolic(k, k_star):
    return 1.0 / (1.0 + k / k_star)


def model_ratio_fid(k, sigma12, k_high, beta, k_low, alpha):
    """
    k: array of wavenumbers
    sigma12: array of sigma12(z) corresponding to each k
    r_nowiggle: ratio P_lin / P_lin_nw
    k_star: computed from sigma12 using hybrid model
    Sig0: overall nonlinear damping factor
    """
    k_star = kstar_stable(sigma12, k_high, beta, k_low, alpha)
    return B_hyperbolic(k, k_star)


# Parameters fit to the fiducial simulations
# These are fit results

RATIO_NL_FIT = dict(k0=0.1394664702344858, alpha=2.0557375287321715)
MODEL_RATIO_FID_FIT = dict(k_high=2.5802, beta=2.5175, k_low=0.0433, alpha=5.2495)

# -----------------------------
# Redshift mapping
# -----------------------------

def find_z_tilde(sigma12_target, A_s_ref, h_ref, omega_cdm_ref,
                 R=12.0, z_min=-0.5, z_max=3.0,
                 xtol=1e-8, rtol=1e-10):

    """
    Find z_tilde such that sigma_12(z_tilde) in the reference cosmology matches sigma12_target.
     - sigma12_target: the target sigma_12 value we want to match
     - A_s_ref, h_ref, omega_cdm_ref: reference cosmological parameters for computing sigma_12
     - R: scale for sigma_12 (default 12 Mpc)
     - z_min, z_max: search range for redshift
    """

    # Request the growth interpolator so compute_sigma12 returns (sigma12_at_z0, D_spline)
    sigma12_z0, D_growth_spline = compute_sigma12(A_s_ref, h_ref, omega_cdm_ref, R=12.0, z=0.0, give_growth=True)

    def sigma_diff(z):
        sigma_12_z = sigma12_z0*D_growth_spline(z)
        return sigma_12_z - sigma12_target

    f_min = sigma_diff(z_min)
    f_max = sigma_diff(z_max)

    if f_min * f_max > 0:
        raise ValueError(
            f"sigma12_target={sigma12_target:.6f} is out of bounds for z in "
            f"[{z_min}, {z_max}] (sigma12 there spans "
            f"[{min(sigma12_z0*D_growth_spline(z_min), sigma12_z0*D_growth_spline(z_max)):.6f}, "
            f"{max(sigma12_z0*D_growth_spline(z_min), sigma12_z0*D_growth_spline(z_max)):.6f}]). "
            f"Widen z_min/z_max."
        )

    z_tilde = brentq(sigma_diff, z_min, z_max, xtol=xtol, rtol=rtol)
    return z_tilde


# ----------------------------
# Vorticity model (with plateau)
# ----------------------------

def Pk_vorticity(k, z, D1, a=2.5, d=1.5, alpha=1.5, b=2.5):
    kp = (1.0 + z)*FIDUCIAL.h
    ks = alpha * kp

    x1 = k / kp
    x2 = k / ks

    P0 = 5.0/(FIDUCIAL.h**3)
    #P1 = 0.6/(h_test**3)
    Pk_ampl = 2.0*P0 * D1**7
    #Pk_ampl = 2.0 * P0 * (P1 / P0)**z * D1**7
    return Pk_ampl * (x1**a) / ((1 + x1**b) * (1 + x2**b)**(d/b))

# -----------------------------
# Nonlinear model
# -----------------------------
# Pk_ref_nl_mod and Pk_ref_nl_fromsim share almost their entire body: both
# map (h, omega_cdm, z) to a fiducial-cosmology z_tilde at fixed sigma_12,
# evaluate the same linear-ratio-based nonlinear correction, then differ
# only in which fiducial nonlinear shape they multiply it against, plus an
# identical derivative-correction / optional-vorticity tail. The shared
# pieces are factored into the two helpers below so a fix only has to be
# made once.

def _pk_ref_nl_prep(k, z, h, omega_cdm, pktheta_fid_interp, pktheta_omega_cdm_interp,
                    A_s_ref, h_ref):
    """Shared setup for both nonlinear P_theta models.
    Returns (k, sigma12_val, D_growth_test_spline, z_tilde, points_tilde, ratio_nl).
    """
    k = np.atleast_1d(k)
    sigma12_val, D_growth_test_spline = compute_sigma12(A_s_ref, h, omega_cdm, R=12.0, z=z, give_growth=True)

    # Compute z_tilde such that in (A_s_ref, h_ref, omega_cdm) cosmology, sigma_12(z_tilde) = sigma12_val
    z_tilde = find_z_tilde(sigma12_val, A_s_ref, h_ref, omega_cdm, R=12.0)
    points_tilde = np.column_stack([k, np.full_like(k, z_tilde)])

    # Model non-linear ratio: get linear ratio from interpolators
    ratio_lin = pktheta_omega_cdm_interp(points_tilde) / pktheta_fid_interp(points_tilde)
    ratio_nl = model_ratio_nl(k, ratio_lin, sigma12_val, **RATIO_NL_FIT)

    return k, sigma12_val, D_growth_test_spline, z_tilde, points_tilde, ratio_nl


def _finalize_pk_theta_nl(Pk_theta_nl, k, z, sigma12_val, D_growth_test_spline,
                          ratio_nl, h, h_ref, deriv, deriv_interp, include_vort):
    """Shared derivative-correction and optional vorticity tail, used by both
    Pk_ref_nl_mod and Pk_ref_nl_fromsim.

    `deriv_interp` may be either:
     - a RegularGridInterpolator that accepts points (sigma, logk)
     - a tuple/list (interp, sigma_grid, logk_grid) where sigma/logk grids
       are provided so we can mask out-of-range logk values
     - None, in which case the derivative interpolator is built lazily
       (expensive) via dPtheta_dh_at_sigma
    """
    if deriv:
        if deriv_interp is None:
            # backward-compatible: compute lazily (expensive)
            dPtheta_dh = dPtheta_dh_at_sigma(k, sigma_fixed=sigma12_val)
        else:
            if isinstance(deriv_interp, (list, tuple)):
                interp, sigma_grid_local, logk_grid_local = deriv_interp
            else:
                interp, sigma_grid_local, logk_grid_local = deriv_interp, None, None

            logk = np.log(k)
            sigma_array = np.full_like(k, sigma12_val)
            pts = np.column_stack([sigma_array, logk])
            dP_vals = interp(pts)

            # If we have logk grid information, zero values outside range
            if logk_grid_local is not None:
                outside_mask = (logk < logk_grid_local[0]) | (logk > logk_grid_local[-1])
                dP_vals = np.array(dP_vals, copy=True)
                dP_vals[outside_mask] = 0.0

            dPtheta_dh = dP_vals

        Pk_theta_nl = Pk_theta_nl + ratio_nl * dPtheta_dh * (h - h_ref)

    if include_vort:
        Pk_vort = Pk_vorticity(k, z, D_growth_test_spline(z))
        Pk_theta_nl = Pk_theta_nl + Pk_vort
        return Pk_theta_nl, Pk_vort

    return Pk_theta_nl


def Pk_ref_nl_mod(k, z, h, omega_cdm,
                    pktheta_fid_interp, pktheta_omega_cdm_interp,
                    A_s_ref, omega_cdm_ref, h_ref, include_vort = False, deriv=True,
                    deriv_interp=None):
    """
    Non-linear velocity power spectrum for a single redshift, using the
    phenomenological fiducial nonlinear ratio (model_ratio_fid) rather than
    an interpolator built directly from simulations. See Pk_ref_nl_fromsim
    for the sim-anchored counterpart.

    Parameters
    ----------
    k : float or array-like
        Wavenumbers in 1/Mpc (NOT h/Mpc -- must match the units
        pktheta_fid_interp / pktheta_omega_cdm_interp were built with).
    z : float
        Redshift (scalar)
    h : float
        Hubble parameter
    omega_cdm : float
        Cold dark matter density parameter
    pktheta_fid_interp : RegularGridInterpolator
        Fiducial interpolator
    pktheta_omega_cdm_interp : RegularGridInterpolator
        Omega_cdm-modified interpolator
    A_s_ref : float
        Reference amplitude

    Returns
    -------
    Pk_theta_nl : float or array
        Non-linear velocity power spectrum
    """
    k, sigma12_val, D_growth_test_spline, z_tilde, points_tilde, ratio_nl = _pk_ref_nl_prep(
        k, z, h, omega_cdm, pktheta_fid_interp, pktheta_omega_cdm_interp, A_s_ref, h_ref)

    # Non-linear damping, evaluated in the fiducial cosmology at z_tilde
    sigma12_val_fid_z0, D_growth_spline_fid = compute_sigma12(A_s_ref, h_ref, omega_cdm_ref, R=12.0, z=0.0, give_growth=True)
    sigma12_val_fid = sigma12_val_fid_z0 * D_growth_spline_fid(z_tilde)
    ratio_fid = model_ratio_fid(k, sigma12_val_fid, **MODEL_RATIO_FID_FIT)

    # Nonlinear P_theta/(Hconf*f)^2 model
    Pk_theta_nl = ratio_nl * ratio_fid * pktheta_fid_interp(points_tilde)

    return _finalize_pk_theta_nl(Pk_theta_nl, k, z, sigma12_val, D_growth_test_spline,
                                 ratio_nl, h, h_ref, deriv, deriv_interp, include_vort)


def Pk_ref_nl_fromsim(k, z, h, omega_cdm,
                         pktheta_fid_interp, pktheta_omega_cdm_interp,
                         pktheta_fid_nl_interp,
                         A_s_ref, omega_cdm_ref, h_ref,
                         include_vort=False, deriv=True,
                         deriv_interp=None):
    """
    Non-linear velocity power spectrum for a single redshift, using a
    fiducial nonlinear interpolator built directly from simulations
    (pktheta_fid_nl_interp, from get_pktheta_nl_interpolator) in place of
    the phenomenological model_ratio_fid used by Pk_ref_nl_mod.

    Parameters
    ----------
    k : float or array-like
        Wavenumbers in 1/Mpc (NOT h/Mpc -- must match the units
        pktheta_fid_interp / pktheta_omega_cdm_interp / pktheta_fid_nl_interp
        were built with).
    z : float
        Redshift (scalar)
    h : float
        Hubble parameter
    omega_cdm : float
        Cold dark matter density parameter
    pktheta_fid_interp : RegularGridInterpolator
        Fiducial linear interpolator
    pktheta_omega_cdm_interp : RegularGridInterpolator
        Omega_cdm-modified linear interpolator
    pktheta_fid_nl_interp : callable
        Fiducial nonlinear interpolator from get_pktheta_nl_interpolator.
        Its `mode` sets what happens outside the trusted k-window of the
        fiducial measurement, [k_min, K_MAX_VALID = 1 /Mpc]:
        mode='strict' returns NaN there (and prints an error), mode='extend'
        converges to linear theory below k_min and continues with a fitted
        power law above k_max. Note that the model is evaluated at the
        mapped redshift z_tilde, so the k-window is the only range check
        that applies here.
    A_s_ref : float
        Reference amplitude

    Returns
    -------
    Pk_theta_nl : float or array
        Non-linear velocity power spectrum
    """
    k, sigma12_val, D_growth_test_spline, z_tilde, points_tilde, ratio_nl = _pk_ref_nl_prep(
        k, z, h, omega_cdm, pktheta_fid_interp, pktheta_omega_cdm_interp, A_s_ref, h_ref)

    # Nonlinear P_theta/(Hconf*f)^2 model, straight from the fiducial sims
    Pk_theta_nl = ratio_nl * pktheta_fid_nl_interp(points_tilde)

    return _finalize_pk_theta_nl(Pk_theta_nl, k, z, sigma12_val, D_growth_test_spline,
                                 ratio_nl, h, h_ref, deriv, deriv_interp, include_vort)


# --------------------------------  
# Interpolator for linear spectra
# --------------------------------

def get_pktheta_interpolator(params, k_min=1e-5, k_max=3e2, nk=500, z_vals=[0.0, 0.1, 0.5, 1.0, 2.0, 3.0]):
    """
    Compute a pktheta interpolator Pk_theta lin(k, z)/(Hconf(z)*f_growth(z))^2 = P_delta_lin(k, z)
    for given cosmology parameters.

    Parameters
    ----------
    params : dict
        Cosmological parameters for Class.
    k_min, k_max : float
        Minimum and maximum k values (h/Mpc)
    nk : int
        Number of k points
    z_vals : list
        Redshifts at which to compute pktheta

    Returns
    -------
    pktheta_interp : RegularGridInterpolator
        Interpolator pktheta_interp([k, z]) -> pktheta
    """
    # Initialize CLASS
    cosmo = Class()
    cosmo.set(params)
    cosmo.set({
        'output':'mPk, dTk, vTk',
        'lensing':'no',
        'z_pk': ','.join(str(z) for z in z_vals),
        'P_k_max_h/Mpc': k_max
    })
    cosmo.compute()
    
    k_arr = np.logspace(np.log10(k_min), np.log10(k_max/2), nk)
    z_arr = np.array(z_vals)
    
    # Prepare 2D array: pktheta[k_index, z_index]
    pktheta_vals = np.zeros((nk, len(z_arr)))
    
    for iz, z in enumerate(z_arr):
        pk_lin = np.array([cosmo.pk_lin(k, z) for k in k_arr])
        pktheta_vals[:, iz] = pk_lin 
    
    cosmo.struct_cleanup()
    cosmo.empty()
    
    # Use RegularGridInterpolator for 2D interpolation
    pktheta_interp = RegularGridInterpolator((k_arr, z_arr), pktheta_vals, bounds_error=False, fill_value=None)
    
    return pktheta_interp

# =====================================================================
# Fiducial nonlinear interpolator built from the fiducial simulations
# =====================================================================
#
# The measured fiducial P_theta(k) is only trustworthy inside a finite
# k-window:
#
#   k < k_min : the box fundamental mode. Below the smallest sampled k
#               there is simply no measurement (and the first few bins
#               contain very few modes, so they are noisy too).
#   k > k_max : shot noise / resolution. K_MAX_VALID = 1 /Mpc is the rough
#               edge of the reliable regime for these runs.
#
# Two treatments of that window are provided, selected with `mode`:
#
#   mode="strict"  Evaluate only inside [k_min, k_max]. Requests outside
#                  return NaN and an explicit error message is printed
#                  (with the offending range), so an out-of-range call can
#                  never silently pollute a figure or a fit.
#
#   mode="extend"  Produce a controlled continuation outside the window,
#                  built *only* from the reliable k < K_MAX_VALID data:
#                    - k < k_min : smoothly converge to the CLASS linear
#                      spectrum (geometric blend over a log-k window; below
#                      the window the output is exactly linear theory).
#                    - k > k_max : power-law tail P ∝ k^n_eff(z), with
#                      n_eff(z) least-squares fitted in log-log over the
#                      reliable sub-range [k_fit_lo, k_max] and anchored on
#                      the last retained node, so the result is continuous
#                      and monotonic rather than following the noisy tail.

K_MAX_VALID = 1.0        # 1/Mpc -- rough edge of the reliable sim range
K_FIT_FRACTION = 3.0     # power-law tail fitted over [k_max / this, k_max]
K_BLEND_FRACTION = 3.0   # low-k blend runs over [k_min / this, k_min]


def _smoothstep(x):
    """C^1 ramp from 0 to 1 on [0, 1] (clipped outside)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _fit_loglog_slope(k, P, k_lo, k_hi):
    """Least-squares log-log slope of P(k) over the window [k_lo, k_hi].

    Used for the high-k power-law continuation: the slope is measured on
    the reliable part of the spectrum only, never on the noisy tail.
    """
    sel = (k >= k_lo) & (k <= k_hi) & (P > 0.0)
    if sel.sum() < 2:
        raise ValueError(
            f"power-law fit window [{k_lo:.4g}, {k_hi:.4g}] 1/Mpc contains "
            f"{sel.sum()} usable nodes (need >= 2); lower k_fit_lo."
        )
    return float(np.polyfit(np.log(k[sel]), np.log(P[sel]), 1)[0])


def get_pktheta_nl_interpolator(base_fiducial=base_fiducial,
                                h=h_fid,
                                redshifts=redshifts,
                                A_s_ref=A_s,
                                mode="extend",
                                kmax=K_MAX_VALID,
                                kmin=None,
                                pktheta_lin_interp=None,
                                k_blend_lo=None,
                                k_blend_hi=None,
                                k_fit_lo=None,
                                loglog=True,
                                verbose=True):
    """
    Interpolator for the *nonlinear* fiducial velocity-divergence spectrum,
    measured from the fiducial simulations, in the same normalised units as
    get_pktheta_interpolator:

        Ptilde_theta(k, z) = Pk_theta_sim(z) / (Hconf(z) * f(z))^2 / h^3   [Mpc^3]
        k[1/Mpc]           = k_sim[h/Mpc] * h

    Drop-in replacement for  model_ratio_fid(k, sigma12) * pktheta_fid_interp
    in Pk_ref_nl_fromsim. Fiducial cosmology only.

    Range of validity
    -----------------
    The interpolation nodes are the sim nodes with kmin <= k <= kmax; `kmax`
    defaults to K_MAX_VALID = 1 /Mpc, the rough edge of the reliable regime.
    Everything outside is handled according to `mode`:

    mode="strict"
        Only k in [k_min, k_max] is evaluated. Out-of-range points return NaN
        and an error message is printed naming the offending k-range and how
        many points were affected. Use this when validating the model against
        the sims, so that no extrapolated value is ever plotted or fitted.

    mode="extend"
        Out-of-range points get a controlled continuation built only from
        k < kmax data:
          - k < k_min: geometric blend onto the CLASS linear spectrum over
            [k_blend_lo, k_blend_hi]; below k_blend_lo the output is exactly
            linear theory. Requires `pktheta_lin_interp`.
          - k > k_max: power law P(k, z) = P(k_max, z) * (k / k_max)^n_eff(z),
            with n_eff(z) fitted in log-log over [k_fit_lo, k_max] (the
            reliable sub-range) at each snapshot redshift and linearly
            interpolated in z. Continuous at k_max by construction.

    Parameters
    ----------
    base_fiducial : str
        Folder with fiducial gevolution outputs (lcdm_box128_pkNNN_theta.dat).
    h : float
        Fiducial Hubble parameter (k conversion h/Mpc -> 1/Mpc and the h^3).
    redshifts : dict
        {'pkNNN': z} snapshot-tag -> redshift (fiducial snapshots are AT these z).
    A_s_ref : float
        Fiducial scalar amplitude, for the CLASS background (H, f).
    mode : {'extend', 'strict'}
        Out-of-range treatment, see above.
    kmax : float
        Upper edge of the trusted range, in 1/Mpc. Default K_MAX_VALID.
    kmin : float or None
        Lower edge of the trusted range, in 1/Mpc. None -> smallest sim node.
    pktheta_lin_interp : callable or None
        Linear interpolator from get_pktheta_interpolator, called as
        pktheta_lin_interp(column_stack([k, z])). Required for mode='extend'.
    k_blend_lo, k_blend_hi : float or None
        Low-k blend window in 1/Mpc. Defaults: k_blend_hi = k_min,
        k_blend_lo = k_min / K_BLEND_FRACTION, i.e. the blend happens entirely
        below the data. Raise k_blend_hi to also fade out the noisiest
        large-scale bins (the first few have very few modes).
    k_fit_lo : float or None
        Lower edge of the power-law fit window. Default k_max / K_FIT_FRACTION.
    loglog : bool
        True (recommended): interpolate log10(P) over (log k, z); smooth and
        positive by construction. False: interpolate P over (k, z).
    verbose : bool
        Print a one-line summary of the configured range at build time, and
        (mode='strict') the out-of-range error messages at call time.

    Returns
    -------
    callable
        interp(points), points = np.column_stack([k_1overMpc, z]) -> P in Mpc^3.
        Attributes for range checks / plotting:
        .mode, .k_min, .k_max (retained data range), .kmax_valid, .k_grid,
        .z_grid, .k_grid_full, .n_eff (fitted tail slopes per z, extend only).
    """
    if mode not in ("strict", "extend"):
        raise ValueError(f"mode must be 'strict' or 'extend', got {mode!r}.")

    # --- read fiducial sims (module's own reader / unit convention) ---
    results = read_gevolution_outputs(base_fiducial, h, redshifts=redshifts)
    z_arr = np.array(sorted(results.keys()))              # ascending

    # --- one CLASS run in the fiducial cosmology; query H(z), f(z) per snapshot ---
    cosmo = Class()
    cosmo.set({
        'h': h, 'omega_b': omega_b, 'omega_cdm': omega_cdm,
        'k_pivot': k_pivot, 'A_s': A_s_ref, 'n_s': n_s,
        'T_cmb': T_cmb, 'N_ur': N_ur,
        'output': 'mPk',
        'z_pk': ','.join(f'{z}' for z in z_arr),
        'P_k_max_h/Mpc': 300.0,
        'non_linear': 'no',
    })
    cosmo.compute()

    k_raw = results[z_arr[0]]['k_theta']                  # h/Mpc (z-independent binning)
    k_full = k_raw * h                                    # 1/Mpc
    P_full = np.zeros((len(k_full), len(z_arr)))
    for iz, z in enumerate(z_arr):
        assert np.allclose(results[z]['k_theta'], k_raw), \
            f"theta k-binning changes at z={z}; cannot build a regular grid."
        Hconf = cosmo.Hubble(z) / (1.0 + z) / h           # == compute_class_background
        f_growth = cosmo.scale_independent_growth_factor_f(z)
        P_full[:, iz] = results[z]['Pk_theta'] / (Hconf * f_growth) ** 2 / h ** 3
    cosmo.struct_cleanup()
    cosmo.empty()

    # --- restrict the nodes to the trusted window ---
    k_lo_req = k_full[0] if kmin is None else kmin
    keep = (k_full >= k_lo_req) & (k_full <= kmax)
    if keep.sum() < 2:
        raise ValueError(
            f"trusted window [{k_lo_req:.4g}, {kmax:.4g}] 1/Mpc keeps "
            f"{keep.sum()} sim nodes (need >= 2); sim range is "
            f"[{k_full[0]:.4g}, {k_full[-1]:.4g}] 1/Mpc."
        )
    k_grid, P = k_full[keep], P_full[keep, :]
    k_min_data, k_max_data = k_grid[0], k_grid[-1]

    # --- core interpolator on the trusted nodes ---
    logk = np.log(k_grid)
    if loglog:
        _rgi = RegularGridInterpolator((logk, z_arr), np.log10(np.clip(P, 1e-30, None)),
                                       bounds_error=False, fill_value=None)
        _nl = lambda kk, zz: 10.0 ** _rgi(np.column_stack([np.log(kk), zz]))
    else:
        _rgi = RegularGridInterpolator((k_grid, z_arr), P,
                                       bounds_error=False, fill_value=None)
        _nl = lambda kk, zz: _rgi(np.column_stack([kk, zz]))

    # =================================================================
    # mode = 'strict': evaluate inside the window only, complain outside
    # =================================================================
    if mode == "strict":
        def interp(points):
            points = np.atleast_2d(np.asarray(points, dtype=float))
            kk, zz = points[:, 0], points[:, 1]
            inside = (kk >= k_min_data) & (kk <= k_max_data)
            out = np.full(kk.shape, np.nan)
            if inside.any():
                out[inside] = np.asarray(_nl(kk[inside], zz[inside])).ravel()
            n_bad = int((~inside).sum())
            if n_bad and verbose:
                bad = kk[~inside]
                print(
                    f"ERROR [get_pktheta_nl_interpolator, mode='strict']: "
                    f"{n_bad}/{kk.size} requested k values lie outside the "
                    f"validity range [{k_min_data:.4g}, {k_max_data:.4g}] 1/Mpc "
                    f"(offending k in [{bad.min():.4g}, {bad.max():.4g}] 1/Mpc); "
                    f"these are returned as NaN. Restrict the k array, or use "
                    f"mode='extend' for a controlled continuation."
                )
            return np.clip(out, 0.0, None)

        n_eff = None
        ratio_edge = None
        blend_warning = None

    # =================================================================
    # mode = 'extend': linear theory below, fitted power law above
    # =================================================================
    else:
        if pktheta_lin_interp is None:
            raise ValueError(
                "mode='extend' needs pktheta_lin_interp (the CLASS linear "
                "interpolator from get_pktheta_interpolator) to converge to "
                "linear theory below k_min. Pass it, or use mode='strict'."
            )

        blend_warning = None
        khi = k_min_data if k_blend_hi is None else k_blend_hi
        klo = k_min_data / K_BLEND_FRACTION if k_blend_lo is None else k_blend_lo
        if not (klo < khi):
            raise ValueError("need k_blend_lo < k_blend_hi.")

        # high-k tail: fit n_eff(z) on the reliable sub-range only
        kfit = k_max_data / K_FIT_FRACTION if k_fit_lo is None else k_fit_lo
        n_eff = np.array([_fit_loglog_slope(k_grid, P[:, iz], kfit, k_max_data)
                          for iz in range(len(z_arr))])

        # Consistency check at the top of the blend window: the blend is only
        # seamless if the measured and linear spectra actually agree there.
        k_edge = np.full_like(z_arr, khi)
        ratio_edge = (np.asarray(_nl(k_edge, z_arr)).ravel()
                      / np.asarray(pktheta_lin_interp(
                          np.column_stack([k_edge, z_arr]))).ravel())
        if np.max(np.abs(ratio_edge - 1.0)) > 0.05:
            blend_warning = (
                f"WARNING [get_pktheta_nl_interpolator]: sim and linear spectra "
                f"differ by up to {100 * np.max(np.abs(ratio_edge - 1.0)):.1f}% at "
                f"the top of the blend window (k = {khi:.4g} 1/Mpc), so the "
                f"low-k blend will show a kink. Lower k_blend_hi/k_blend_lo, or "
                f"check the normalisation of the fiducial run."
            )

        def _tail(kk, zz):
            """Power-law continuation above k_max_data, anchored on the data."""
            slope = np.interp(zz, z_arr, n_eff)          # clamped outside z-grid
            P_anchor = np.asarray(_nl(np.full_like(kk, k_max_data), zz)).ravel()
            return P_anchor * (kk / k_max_data) ** slope

        def interp(points):
            points = np.atleast_2d(np.asarray(points, dtype=float))
            kk, zz = points[:, 0], points[:, 1]

            out = np.asarray(_nl(np.clip(kk, None, k_max_data), zz)).ravel()

            # --- above the trusted range: fitted power law ---
            hi = kk > k_max_data
            if hi.any():
                out[hi] = _tail(kk[hi], zz[hi])

            # --- below the trusted range: converge to linear theory ---
            Plin = np.asarray(pktheta_lin_interp(np.column_stack([kk, zz]))).ravel()
            w = _smoothstep((np.log(kk) - np.log(klo)) / (np.log(khi) - np.log(klo)))
            blend = np.exp(w * np.log(np.clip(out, 1e-30, None))
                           + (1.0 - w) * np.log(np.clip(Plin, 1e-30, None)))
            out = np.where(kk <= klo, Plin, np.where(kk >= khi, out, blend))

            return np.clip(out, 0.0, None)

    interp.mode = mode
    interp.k_min = k_min_data          # 1/Mpc, retained data range
    interp.k_max = k_max_data
    interp.kmax_valid = kmax           # requested validity ceiling
    interp.k_grid = k_grid             # 1/Mpc, retained nodes
    interp.z_grid = z_arr
    interp.k_grid_full = k_full        # full sim k-range before the cut
    interp.n_eff = n_eff               # fitted tail slopes per z (extend only)
    interp.ratio_edge = ratio_edge     # P_sim / P_lin at k_blend_hi, per z

    if verbose:
        msg = (f"[get_pktheta_nl_interpolator] mode='{mode}', "
               f"{keep.sum()}/{len(k_full)} sim nodes kept, "
               f"valid range [{k_min_data:.4g}, {k_max_data:.4g}] 1/Mpc")
        if mode == "extend":
            msg += (f"; linear blend on [{klo:.4g}, {khi:.4g}], "
                    f"tail slope n_eff in [{n_eff.min():.2f}, {n_eff.max():.2f}] "
                    f"fitted on [{kfit:.4g}, {k_max_data:.4g}]")
        print(msg)
        if blend_warning:
            print(blend_warning)

    return interp
