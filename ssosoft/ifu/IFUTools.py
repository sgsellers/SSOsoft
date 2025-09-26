import warnings
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as nrec
import numpy.polynomial.polynomial as npoly
import numpy.typing as npt
import scipy.integrate as scint
import scipy.interpolate as scinterp
import scipy.ndimage as scind
import scipy.optimize as scopt
import scipy.signal as scig
import tqdm
from astropy.constants import c
from matplotlib.widgets import Slider

from ..spectral import spectraTools as spex

c_kms = c.value / 1e3

"""
This file contains generalized helper functions for IFU reductions. Many of these are
contained in the ssosoft.spectral.spectraTools module. These implementations have been
adjusted to correspond to IFU processing.
"""

def create_gaintables(
        flat: np.ndarray, gain_line: float, fiber_map: np.ndarray,
        neighborhood: int=5, edge_padding: int=100
    ) -> tuple[np.ndarray, np.ndarray]:
    """Generates a set of gain tables given a starting set of initial parameters.
    Attempts two sequential refinements of gain, which will be applied to the flat
    field before moving on to the spectral continuum fitting:
    1.) Coarse gain using the mean profile of the center neighborhood living fibers.
        Standard shift-and-divide method.
    2.) Fine gain using surrounding fiber-average profile
        Standard shift-and-divide method using initial skews to get close.
        This step also fine-tunes to skew along the slit.

    Parameters
    ----------
    flat : np.ndarray
        Dark-corrected, rigidly-deskewed flat field
    gain_line : float
        Index of the gain line.
    fiber_map : np.ndarray
        1D fiber mask map of shape (5, n_fibers)
    neighborhood : int, optional
        Number of surrounding fibers to average in fine gain, by default 5
    edge_padding : int, optional
        Range to clip profiles to in pixels for fine gain table creation

    Returns
    -------
    coarse_gain : np.ndarray
        Coarse gain table using single reference profile shift-and-divide
    fine_gain : np.ndarray
        Fine gain table using nearest neighbor profiles shift-and-divide
    """
    center_mean_profile = np.zeros(flat.shape[1])
    good_fiber_counter = 0
    row_counter = 0
    while good_fiber_counter < neighborhood:
        fiber_xrange = fiber_map[:, fiber_map.shape[1] // 2 - 3 + row_counter].round().astype(int)
        if not all(fiber_xrange == 0): # Dead fiber
            center_mean_profile += np.sum(flat[fiber_xrange], axis=0)
            good_fiber_counter += 1
        row_counter += 1
    center_mean_profile /= np.median(center_mean_profile)
    profile_linecore = spex.find_line_core(
        center_mean_profile[int(gain_line) - 9:int(gain_line) + 11]
    ) + int(gain_line) - 9
    shifted_lines = np.ones(flat.shape)
    for i in range(flat.shape[0]):
        if i == 0 or i not in fiber_map.round().astype(int):
            # Do not attempt to shift to fibers outside bundle
            continue
        line_position = spex.find_line_core(
            flat[i, int(gain_line - 9):int(gain_line + 11)]
        ) + int(gain_line - 9)
        shift = line_position - profile_linecore
        shifted_lines[i, :] = scind.shift(center_mean_profile, shift, mode="nearest")
    coarse_gaintable = flat / shifted_lines

    # Smooth rough gaintable in the chosen line
    if gain_line < 20:
        lowidx = 0
    else:
        lowidx = int(gain_line - 20)
    if gain_line > flat.shape[0] - 20:
        highidx = int(flat.shape[0] - 1)
    else:
        highidx = int(gain_line + 20)
    for i in range(coarse_gaintable.shape[1]):
        coarse_gaintable[int(gain_line - 7):int(gain_line + 7), i] = np.nanmean(coarse_gaintable[lowidx:highidx, i])
    coarse_gaintable /= np.nanmedian(coarse_gaintable)

    corrected_flat = flat / coarse_gaintable

    shifted_lines = np.ones(corrected_flat.shape)
    mean_profiles = np.zeros((flat.shape[1], fiber_map.shape[1]))
    for i in range(fiber_map.shape[1]):
        mean_profile = np.zeros(flat.shape[1])
        good_fiber_counter = 0
        row_counter = i - neighborhood // 2 if i - neighborhood // 2 > 0 else 0
        if row_counter > fiber_map.shape[1] - neighborhood:
            row_counter = fiber_map.shape[1] - neighborhood
        while good_fiber_counter < neighborhood:
            if row_counter == fiber_map.shape[1]:
                break
            fiber_xrange = fiber_map[:, row_counter].round().astype(int)
            if not all(fiber_xrange == 0): # Dead fiber
                mean_profile += np.sum(corrected_flat[fiber_xrange], axis=0)
                good_fiber_counter += 1
            row_counter += 1
        mean_profiles[:, i] = mean_profile / np.median(mean_profile)

    for i in range(fiber_map.shape[1]):
        if all(fiber_map[:, i] == 0):
            continue
        fiber_xrange = fiber_map[:, i].round().astype(int)
        for j in fiber_xrange:
            ref_profile = corrected_flat[j, :] / np.nanmedian(corrected_flat[j, :])
            line_shift = iterate_shifts(
                ref_profile[edge_padding:-edge_padding],
                mean_profiles[edge_padding:-edge_padding, i]
            )
            shifted_lines[j] = scind.shift(mean_profiles[:, i], line_shift, mode="nearest")
    gaintable = flat / shifted_lines
    gaintable /= np.nanmedian(gaintable)
    return gaintable, coarse_gaintable

def iterate_shifts(reference_profile: np.ndarray, mean_profile: np.ndarray, nzones: int=5) -> float:
    """
    Determines best shift for the mean profile to the reference profile from the median shift in each of N zones

    Parameters
    ----------
    reference_profile: numpy.ndarray
        Profile to determine shifts to
    mean_profile: numpy.ndarray
        Profile to shift
    nzones: int
        Number of subfields to consider shifts for

    Returns
    -------
    float
        Median of each subfield shift
    """
    reference_slices = np.array_split(reference_profile, nzones)
    mean_slices = np.array_split(mean_profile, nzones)
    shifts = np.zeros(len(mean_slices))
    for i in range(len(reference_slices)):
        shifts[i] = scopt.minimize_scalar(
            fit_profile,
            bounds=(-5, 5),
            args=(reference_slices[i], mean_slices[i])
        ).x
    return np.nanmedian(shifts)

def fit_profile(shift: float, reference_profile: np.ndarray, mean_profile: np.ndarray, landing_width: int=5) -> float:
    """
    Alternate minimization of shift residuals using the final "gain" image

    Parameters
    ----------
    shift: float
        Value for shift
    reference_profile: numpy.ndarray
        Reference profile to divide against
    mean_profile: numpy.ndarray
        Mean Profile for division
    landing_width: int
        determines slope/bg of residuals. Higher to negate edge effects

    Returns
    -------
    fit_metric: float
        Sum of "gain" profile
    """
    shifted_mean = scind.shift(mean_profile, shift, mode="nearest")
    divided = reference_profile / shifted_mean
    slope = (np.nanmean(divided[-landing_width:]) - np.nanmean(divided[:landing_width])) / divided.size
    bg = slope * np.arange(divided.size) + np.nanmean(divided[:landing_width])
    gainsub = np.abs(divided - bg)
    fit_metric = np.nansum(gainsub[np.isfinite(gainsub)])
    return fit_metric

def spectral_gain(
        flat: np.ndarray, wavelength_grid: np.ndarray, fiber_map: np.ndarray,
        fts_wavelengths: np.ndarray, fts_spectrum: np.ndarray,
        edge_padding: int=100, smoothing=50, full=False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a spectral gain table to correct for variations in illumination
    across the spectral range observed. This is done by fitting the continuum
    points with an nth order polynomial, where n is on the range(1, 10).

    Continuum points are determined from the FTS atlas spectrum.
    Regions with spectral values above 0.95 are preferred, but in the event
    that there is not an adequate sampling across the field, the brightest 10%
    of pixels in each of the image_sectors of the spectral range will be used.

    Parameters
    ----------
    flat : np.ndarray
        Deskewed and gain-corrected flat field
    wavelength_grid : np.ndarray
        Fine-adjusted wavelength grid from FTS comparison
    fiber_map : np.ndarray
        Shape (5, 400) map of fiber positions
    fts_wavelengths : np.ndarray
        Non-degraded wavelengths of the corresponding FTS window
    fts_spectrum : np.ndarray
        Non-degraded spectral values of the corresponding FTS window
    edge_padding : int, optional
        Number of points to clip from the edges of the spectrum for continuum finding.
        By default 100
    smoothing : int, optional
        Window for median-filtering FTS/reference spectral residuals
    full : bool, optional
        If True, returns the residual values and order of polynomial fit used

    Returns
    -------
    spectral_gain : np.ndarray
        Spectral gain table for normalization across the field
    residuals : np.ndarray, optional
        If full=True, returns an array with polynomial fit residuals
    pfit_orders : np.ndarray, optional
        If full=True, returns an array with polynomial order used in fitting
    """
    fts_spectrum_interpolated = scind.gaussian_filter(
        np.interp(wavelength_grid, fts_wavelengths, fts_spectrum),
        2
    )
    spectral_gain = np.zeros(flat.shape)
    residuals = np.zeros(flat.shape[0])
    pfit_orders = np.zeros(flat.shape[0])
    for i in range(fiber_map.shape[1]):
        if all(fiber_map[:, i]) == 0:
            continue
        fiber_xrange = fiber_map[:, i].round().astype(int)
        for j in fiber_xrange:
            ref_profile = flat[j]
            divd_profile = ref_profile / fts_spectrum_interpolated
            smoothed_profile = scind.median_filter(divd_profile, smoothing)
            fit_range = smoothed_profile[edge_padding:-edge_padding]
            prev_iter_resid = 0
            polyfit = np.ones(wavelength_grid.shape[0])
            for order in range(2, 4):
                pfit, diags = npoly.Polynomial.fit(
                    wavelength_grid[edge_padding:-edge_padding],
                    fit_range,
                    order,
                    full=True
                )
                if order != 2 and diags[0] > prev_iter_resid:
                    # Worse fit, break
                    break
                prev_iter_resid = diags[0]
            spectral_gain[j] = pfit(wavelength_grid)
            residuals[j] = prev_iter_resid
            pfit_orders[j] = order
    gain_mask = spectral_gain == 0
    spectral_gain /= np.nanmedian(spectral_gain[~gain_mask])
    spectral_gain[gain_mask] = 1
    if full:
        return spectral_gain, residuals, pfit_orders
    return spectral_gain


# fts_splits = np.array_split(fts_spectrum_interpolated, image_sectors) # List of arrays
#     quadrant_masks = []
#     for sector in fts_splits:
#         mask = sector >= 0.85 # avoid line cores
#         # Less than 10% continuum:
#         if len(sector[mask]) < 0.1 * sector.shape[0]:
#             threshold_value = np.percentile(sector, 85)
#             mask = sector >= threshold_value
#         quadrant_masks.append(mask)

#     master_mask = np.concatenate(quadrant_masks)
#     masked_wavelengths = wavelength_grid[edge_padding:-edge_padding][master_mask]

#     masked_fts = fts_spectrum_interpolated[master_mask]

    # prev_iter_resid = 0
    # for order in range(1, 6):
    #     pfit, diags = npoly.Polynomial.fit(masked_wavelengths, masked_fts, order, full=True)
    #     if order != 1 and diags[0] > prev_iter_resid:
    #         # Worse fit, break immediately
    #         break
    #     prev_iter_resid = diags[0]
    #     fts_fit = np.zeros(wavelength_grid.shape[0])
    #     for k, exp in enumerate(pfit):
    #         fts_fit += exp * wavelength_grid ** k

    ### This is incorrect and performs worse than the regular gain.
    ### What we should do is divide the profile by the interpolated FTS
    ### profile, and fit THAT with an ND polynomial after smoothing. THAT's the gain

    # spectral_gain = np.ones(flat.shape)
    # max_order = []
    # coeffs = []
    # for i in range(fiber_map.shape[1]):
    #     if all(fiber_map[:, i] == 0):
    #         continue
    #     fiber_xrange = fiber_map[:, i].round().astype(int)
    #     for j in fiber_xrange:
    #         ref_profile = (flat[j, :] / np.nanmedian(flat[j, :]))[edge_padding:-edge_padding]
    #         masked_profile = ref_profile[master_mask]
    #         prev_iter_resid = 0
    #         polyfit = np.ones(wavelength_grid.shape[0])
    #         ord = 0
    #         # Polynomial orders
    #         for order in range(2, 5):
    #             pfit, diags = npoly.Polynomial.fit(masked_wavelengths, masked_profile, order, full=True)
    #             if order != 2 and diags[0] > prev_iter_resid:
    #                 # Worse fit, break immediately
    #                 break
    #             prev_iter_resid = diags[0]
    #             polyfit = np.zeros(wavelength_grid.shape[0])
    #             coeffs.append(pfit)
    #             for k, exp in enumerate(pfit):
    #                 polyfit += exp * wavelength_grid ** k
    #             ord = order
    #         spectral_gain[j] = polyfit / fts_fit
    #         max_order.append(ord)
    # spectral_gain /= np.nanmedian(spectral_gain[spectral_gain != spectral_gain[0, 0]])
    # spectral_gain[spectral_gain == spectral_gain[0, 0]] = 1

    return spectral_gain, max_order, coeffs, master_mask
