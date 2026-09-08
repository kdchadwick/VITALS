"""
spectral_utils.py
=================

Lightweight spectral-analysis helpers for the SEG critical-minerals tutorial.

The NASA source repositories this tutorial is built on (EMIT-Data-Resources,
VITALS, mined-lands, ornl-airborne) rely on the pre-computed EMIT L2B mineralogy
(Tetracorder) product for mineral identification and do **not** ship code that
computes continuum removal or absorption-feature metrics from raw reflectance.
This module fills that gap so the tutorial can demonstrate, on both EMIT and
AVIRIS-5 reflectance, *why* a given pixel is identified as alunite versus white
mica -- i.e. the diagnostic Al-OH absorption features that drive the
identification (after Clark & Roush, 1984; Portela et al., 2025).

Everything here works on plain 1-D numpy arrays of wavelength (micrometres) and
reflectance, so it is agnostic to whether the spectrum came from EMIT (xarray
`wavelengths`, plural) or AVIRIS-5 (`wavelength`, singular).

Functions
---------
to_micrometers        : normalize a wavelength array to micrometres.
subset_range          : slice a (wavelength, reflectance) pair to a window.
continuum_removed     : convex-hull-free straight-line continuum removal.
band_depth            : 1 - min(continuum-removed) over a feature window.
minimum_wavelength    : parabola-refined wavelength of an absorption minimum.
feature_width_fwhm    : full-width at half the maximum feature depth.
aloh_feature_metrics  : convenience wrapper returning position/depth/width for
                        the 2.1-2.25 um Al-OH feature.
DIAGNOSTIC_FEATURES   : dict of textbook diagnostic absorption positions.
diagnostic_lookup     : map a USGS-splib entry name to its feature description.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Diagnostic absorption features for the advanced-argillic / sericitic suite.
# Positions are in micrometres and are the textbook / USGS-spectral-library
# values used to interpret continuum-removed spectra in this tutorial.
# Primary Al-OH position is the feature this tutorial keys on.
# ---------------------------------------------------------------------------
DIAGNOSTIC_FEATURES = {
    "alunite": {
        "primary_AlOH_um": (2.160, 2.170),   # broad Al-OH
        "secondary_um": 2.320,               # secondary alunite feature
        "note": "Broad Al-OH ~2.16-2.17 um; advanced argillic (high-sulfidation).",
    },
    "pyrophyllite": {
        "primary_AlOH_um": (2.165, 2.170),   # sharp, narrow
        "secondary_um": 2.320,
        "note": "Sharp/narrow Al-OH ~2.166 um; feature narrower than alunite.",
    },
    "kaolinite": {
        "primary_AlOH_um": (2.160, 2.210),   # doublet
        "doublet_um": (2.160, 2.206),
        "note": "Al-OH doublet ~2.16/2.20 um; well-crystalline kaolinite.",
    },
    "dickite": {
        "primary_AlOH_um": (2.170, 2.210),
        "doublet_um": (2.175, 2.208),
        "note": "Kaolin-group; doublet shifted slightly longward of kaolinite.",
    },
    "muscovite_white_mica": {
        "primary_AlOH_um": (2.185, 2.215),   # shifts with Al content / fluid pH
        "secondary_um": 2.350,
        "note": "Al-OH ~2.19-2.21 um; position shifts with octahedral Al "
                "(short=Al-rich/phengite-poor, long=Al-poor/phengitic) -> pH proxy.",
    },
    "illite": {
        "primary_AlOH_um": (2.200, 2.210),
        "note": "White-mica group; broadly similar to muscovite ~2.20 um.",
    },
}

# Default analysis window for the Al-OH feature used across the tutorial (um).
ALOH_WINDOW_UM = (2.10, 2.26)


def to_micrometers(wavelengths):
    """Return wavelengths in micrometres.

    EMIT and AVIRIS both report wavelength in nanometres; if the input looks
    like nanometres (max > 100) it is divided by 1000.
    """
    w = np.asarray(wavelengths, dtype="float64")
    if np.nanmax(w) > 100.0:
        w = w / 1000.0
    return w


def _valid(wavelengths, reflectance):
    """Sort by wavelength and drop NaN / non-finite samples."""
    w = np.asarray(wavelengths, dtype="float64")
    r = np.asarray(reflectance, dtype="float64")
    order = np.argsort(w)
    w, r = w[order], r[order]
    good = np.isfinite(w) & np.isfinite(r)
    return w[good], r[good]


def subset_range(wavelengths, reflectance, wl_range):
    """Slice a (wavelength, reflectance) pair to ``wl_range`` (in same units)."""
    w, r = _valid(wavelengths, reflectance)
    lo, hi = wl_range
    m = (w >= lo) & (w <= hi)
    return w[m], r[m]


def continuum_removed(wavelengths, reflectance, wl_range=None):
    """Straight-line (single-segment) continuum removal.

    A straight line ("continuum" or "hull") is drawn between the reflectance at
    the two ends of the window and the spectrum is divided by that line. The
    result is 1.0 on the shoulders and dips toward 0 in an absorption band --
    the standard way to isolate a diagnostic feature (Clark & Roush, 1984).

    Parameters
    ----------
    wavelengths, reflectance : array-like
        Full spectrum. Units of ``wavelengths`` and ``wl_range`` must match.
    wl_range : (lo, hi), optional
        Feature window. Defaults to :data:`ALOH_WINDOW_UM` (assumes micrometres).

    Returns
    -------
    w, cr : ndarray
        Wavelengths in the window and the continuum-removed reflectance.
    """
    if wl_range is None:
        wl_range = ALOH_WINDOW_UM
    w, r = subset_range(wavelengths, reflectance, wl_range)
    if w.size < 3:
        return w, np.full_like(w, np.nan)
    # Endpoint continuum line between first and last sample in the window.
    slope = (r[-1] - r[0]) / (w[-1] - w[0])
    continuum = r[0] + slope * (w - w[0])
    # Guard against zero/negative continuum.
    continuum = np.where(continuum <= 0, np.nan, continuum)
    cr = r / continuum
    return w, cr


def band_depth(wavelengths, reflectance, wl_range=None):
    """Maximum band depth = 1 - min(continuum-removed reflectance)."""
    _, cr = continuum_removed(wavelengths, reflectance, wl_range)
    if np.all(np.isnan(cr)):
        return np.nan
    return float(1.0 - np.nanmin(cr))


def minimum_wavelength(wavelengths, reflectance, wl_range=None, refine=True):
    """Wavelength of the deepest point of the continuum-removed feature.

    With ``refine=True`` a parabola is fit to the three samples around the
    discrete minimum to recover a sub-band-spacing feature position -- this is
    what lets a ~7.4 nm (EMIT) or ~7 nm (AVIRIS) sampled spectrum resolve the
    small Al-OH position shifts that distinguish alunite (~2.16 um) from
    Al-rich vs Al-poor white micas (2.19-2.21 um).

    Returns the minimum wavelength in the same units as ``wavelengths``
    (micrometres if you passed micrometres), or NaN if not resolvable.
    """
    w, cr = continuum_removed(wavelengths, reflectance, wl_range)
    if w.size < 3 or np.all(np.isnan(cr)):
        return np.nan
    i = int(np.nanargmin(cr))
    if not refine or i == 0 or i == len(w) - 1:
        return float(w[i])
    # Parabolic (3-point) vertex refinement.
    x0, x1, x2 = w[i - 1], w[i], w[i + 1]
    y0, y1, y2 = cr[i - 1], cr[i], cr[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if not np.isfinite(denom) or denom == 0:
        return float(x1)
    # Vertex offset in units of the (assumed ~uniform) sample spacing.
    delta = 0.5 * (y0 - y2) / denom
    step = (x2 - x0) / 2.0
    return float(x1 + delta * step)


def feature_width_fwhm(wavelengths, reflectance, wl_range=None):
    """Full-width at half maximum depth of the continuum-removed feature.

    Alunite has a broad Al-OH feature; pyrophyllite's is markedly narrower --
    feature width is the parameter Portela et al. (2025) use in a decision tree
    to separate alunite- from pyrophyllite-rich zones. Returned in the same
    units as ``wavelengths``; NaN if the half-depth crossings are not bracketed.
    """
    w, cr = continuum_removed(wavelengths, reflectance, wl_range)
    if w.size < 3 or np.all(np.isnan(cr)):
        return np.nan
    depth = 1.0 - cr
    dmax = np.nanmax(depth)
    if not np.isfinite(dmax) or dmax <= 0:
        return np.nan
    half = dmax / 2.0
    above = depth >= half
    idx = np.where(above)[0]
    if idx.size < 2:
        return np.nan
    # Linear interpolation of the half-depth crossings on each side.
    left, right = idx[0], idx[-1]

    def _cross(i_in, i_out):
        # interpolate wavelength where depth == half between samples i_out,i_in
        d_in, d_out = depth[i_in], depth[i_out]
        if d_in == d_out:
            return w[i_in]
        f = (half - d_out) / (d_in - d_out)
        return w[i_out] + f * (w[i_in] - w[i_out])

    wl_left = _cross(left, left - 1) if left > 0 else w[left]
    wl_right = _cross(right, right + 1) if right < len(w) - 1 else w[right]
    return float(abs(wl_right - wl_left))


def aloh_feature_metrics(wavelengths, reflectance, wl_range=None):
    """Convenience: position, depth and width of the Al-OH feature.

    Returns a dict with ``min_wavelength_um``, ``band_depth`` and
    ``fwhm_um`` (assuming micrometre input). Wavelengths are converted to
    micrometres internally so nanometre input is also accepted.
    """
    w_um = to_micrometers(wavelengths)
    if wl_range is None:
        wl_range = ALOH_WINDOW_UM
    return {
        "min_wavelength_um": minimum_wavelength(w_um, reflectance, wl_range),
        "band_depth": band_depth(w_um, reflectance, wl_range),
        "fwhm_um": feature_width_fwhm(w_um, reflectance, wl_range),
    }


def bad_band_nan(wavelengths, reflectance, drop_ranges_um=None):
    """Set reflectance to NaN inside atmospheric water-vapour / edge windows.

    Default ranges are the standard ~1.4 um and ~1.9 um water-vapour bands plus
    the VSWIR spectral tails (after ornl-airborne AVIRIS notebooks). Input
    wavelength may be nm or um; ``drop_ranges_um`` are in micrometres.
    """
    if drop_ranges_um is None:
        drop_ranges_um = [(0.0, 0.45), (1.34, 1.48), (1.80, 1.98), (2.46, 99.0)]
    w_um = to_micrometers(wavelengths)
    r = np.asarray(reflectance, dtype="float64").copy()
    for lo, hi in drop_ranges_um:
        r[(w_um >= lo) & (w_um <= hi)] = np.nan
    return r


def diagnostic_lookup(name):
    """Map a USGS-splib entry name (from the L2B mineral metadata / grouping
    matrix ``Name`` column) to a short diagnostic-feature description.

    Example: ``"Alunite GDS97 K Syn (150C) W2R4Na"`` -> alunite entry.
    Returns ``(mineral_key, feature_dict)`` or ``(None, None)`` if unmatched.
    """
    if not isinstance(name, str):
        return None, None
    n = name.lower()
    # Order matters: check compound / mica names before bare group names.
    if "pyrophyl" in n and "musc" not in n and "alunite" not in n:
        return "pyrophyllite", DIAGNOSTIC_FEATURES["pyrophyllite"]
    if "alunite" in n:
        return "alunite", DIAGNOSTIC_FEATURES["alunite"]
    if "dickite" in n:
        return "dickite", DIAGNOSTIC_FEATURES["dickite"]
    if "kaolin" in n:
        return "kaolinite", DIAGNOSTIC_FEATURES["kaolinite"]
    if "muscovite" in n or "sericite" in n:
        return "muscovite_white_mica", DIAGNOSTIC_FEATURES["muscovite_white_mica"]
    if "illite" in n:
        return "illite", DIAGNOSTIC_FEATURES["illite"]
    return None, None
