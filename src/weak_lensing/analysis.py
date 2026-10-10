'''Measure galaxy shapes, stack tangential profiles, and infer NFW parameters.'''

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import galsim
import numpy as np


def load_galaxy_stamps(filepath: str | Path) -> list[galsim.Image]:
    '''Load galaxy stamp images from a FITS file.
    Args
        filepath (str or Path): Path to the FITS file.
    Returns
        list: List of GalSim Image objects.
    '''
    return galsim.fits.readMulti(filepath)


def get_ellip_unweighted(ims: Sequence[galsim.Image]) -> np.ndarray:
    '''Estimate the ellipticity components for each galaxy image.
    Args
        ims (Sequence): GalSim Image objects in source-catalogue order.
    Returns
        ndarray: Array of shape (n_images, 2), where column 0 contains e1 and
                 column 1 contains e2 = Q_xy / D, where D is the moment denominator.
                 Failed measurements are returned as NaN.
    '''
    ellipticities = []

    for im in ims:
        image = np.asarray(im.array, dtype=float)

        # Pixel coordinate grids
        y, x = np.indices(image.shape)

        # Calculate galaxy centroid
        flux = np.sum(image)
        x_bar = np.sum(image * x) / flux
        y_bar = np.sum(image * y) / flux

        # Coordinates relative to the centroid
        dx = x - x_bar
        dy = y - y_bar

        # Unweighted quadrupole moments
        Q_xx = np.sum(image * dx * dx) / flux
        Q_yy = np.sum(image * dy * dy) / flux
        Q_xy = np.sum(image * dx * dy) / flux

        # Ellipticity denominator
        determinant = Q_xx * Q_yy - Q_xy**2

        if determinant < 0:
            ellipticities.append([np.nan, np.nan])
            continue

        denominator = Q_xx + Q_yy + 2 * np.sqrt(determinant)

        e1 = (Q_xx - Q_yy) / denominator
        e2 = Q_xy / denominator

        ellipticities.append([e1, e2])

    return np.asarray(ellipticities)


def get_ellip_weighted(ims: Sequence[galsim.Image], fwhm: float = 4.0) -> np.ndarray:
    '''Estimate the weighted ellipticity components for each galaxy image.
    Args
        ims (Sequence): GalSim Image objects in source-catalogue order.
        fwhm (float): Full width at half maximum of the circular Gaussian
                      weight function, in pixels.
    Returns
        ndarray: Array of shape (n_images, 2), where column 0 contains e1 and
                 column 1 contains e2 = Q_xy / D, where D is the moment denominator.
                 Failed measurements are returned as NaN.
    '''
    ellipticities = []

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    for im in ims:
        image = np.asarray(im.array, dtype=float)

        # Pixel coordinate grids
        y, x = np.indices(image.shape)

        # Calculate galaxy centroid
        flux = np.sum(image)
        x_bar = np.sum(image * x) / flux
        y_bar = np.sum(image * y) / flux

        # Coordinates relative to the centroid
        dx = x - x_bar
        dy = y - y_bar

        # Circular Gaussian weight function centred on the galaxy centroid
        weight = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

        # Weighted intensity used for the weighted moment normalisation
        weighted_flux = np.sum(image * weight)

        if not np.isfinite(weighted_flux) or weighted_flux == 0:
            ellipticities.append([np.nan, np.nan])
            continue

        # Weighted quadrupole moments
        Q_xx = np.sum(image * weight * dx * dx) / weighted_flux
        Q_yy = np.sum(image * weight * dy * dy) / weighted_flux
        Q_xy = np.sum(image * weight * dx * dy) / weighted_flux

        # Ellipticity denominator
        determinant = Q_xx * Q_yy - Q_xy**2

        if determinant < 0:
            ellipticities.append([np.nan, np.nan])
            continue

        denominator = Q_xx + Q_yy + 2 * np.sqrt(determinant)

        e1 = (Q_xx - Q_yy) / denominator
        e2 = Q_xy / denominator

        ellipticities.append([e1, e2])

    return np.asarray(ellipticities)


def get_ellip_hsm(ims: Sequence[galsim.Image]) -> np.ndarray:
    '''Measure observed galaxy shapes with GalSim HSM adaptive moments.
    Args
        ims (Sequence): GalSim Image objects in source-catalogue order.
    Returns
        ndarray: Array of shape (n_images, 2), where column 0 contains g1 and
                 column 1 contains g2 in the reduced-shear shape convention.
                 These are observed shapes without PSF correction. Failed
                 measurements are returned as NaN.
    '''
    ellipticities = []

    for im in ims:
        try:
            result = galsim.hsm.FindAdaptiveMom(im)

            g1 = result.observed_shape.g1
            g2 = result.observed_shape.g2

            if not np.isfinite(g1) or not np.isfinite(g2):
                ellipticities.append([np.nan, np.nan])
            else:
                ellipticities.append([g1, g2])

        except Exception:
            ellipticities.append([np.nan, np.nan])

    return np.asarray(ellipticities)


def get_tangential_shear_profile(g1: np.ndarray, g2: np.ndarray, source_positions: np.ndarray,
                                 halo_positions: np.ndarray, n_bins: int = 10,
                                 r_min: float = 100.0, r_max: float = 3600.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Calculate the mean tangential shear profile around halo positions.
    Args
        g1 (ndarray): First shear/ellipticity component for each source galaxy.
        g2 (ndarray): Second shear/ellipticity component for each source galaxy.
        source_positions (ndarray): Array of shape (n_sources, 2), containing
                                    source x and y positions in arcseconds.
        halo_positions (ndarray): Array of shape (n_halos, 2), containing halo
                                  x and y positions in arcseconds.
        n_bins (int): Number of logarithmic radial bins.
        r_min (float): Minimum separation in arcseconds.
        r_max (float): Maximum separation in arcseconds.
    Returns
        tuple: Bin centres in arcseconds, mean tangential components, pair
               counts, and sums of squared tangential components; each has
               shape (n_bins,). Empty-bin means are NaN.
    '''
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)

    source_x = source_positions[:, 0]
    source_y = source_positions[:, 1]

    bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    shear_sum = np.zeros(n_bins)
    shear_sum_squared = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    valid_sources = np.isfinite(g1) & np.isfinite(g2)

    for halo_x, halo_y in halo_positions:
        # Separation vector from halo to each source galaxy
        dx = source_x - halo_x
        dy = source_y - halo_y

        # Separation distance and polar angle
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)

        # Tangential shear component for each halo-source pair
        gamma_t = -(g1 * np.cos(2 * phi) + g2 * np.sin(2 * phi))

        valid_pairs = valid_sources & np.isfinite(gamma_t) & (r >= r_min) & (r < r_max)

        bin_index = np.digitize(r[valid_pairs], bin_edges) - 1
        gamma_t_valid = gamma_t[valid_pairs]

        shear_sum += np.bincount(bin_index, weights=gamma_t_valid, minlength=n_bins)
        shear_sum_squared += np.bincount(bin_index, weights=gamma_t_valid**2, minlength=n_bins)

        counts += np.bincount(bin_index, minlength=n_bins)

    mean_gamma_t = np.full(n_bins, np.nan)
    valid_bins = counts > 0
    mean_gamma_t[valid_bins] = shear_sum[valid_bins] / counts[valid_bins]

    return bin_centres, mean_gamma_t, counts, shear_sum_squared


def get_shear_uncertainties(mean_gamma_t: np.ndarray, shear_sum_squared: np.ndarray,
                            counts: np.ndarray) -> np.ndarray:
    '''Estimate uncertainties on the mean tangential shear profile.
    Args
        mean_gamma_t (ndarray): Mean tangential shear in each radial bin.
        shear_sum_squared (ndarray): Sum of gamma_t squared in each radial bin.
        counts (ndarray): Number of halo-source pairs in each radial bin.
    Returns
        ndarray: Standard errors with shape (n_bins,), treating pairs as
                 independent. Bins containing fewer than two pairs are NaN.
    '''
    gamma_t_error = np.full_like(mean_gamma_t, np.nan, dtype=float)

    valid_bins = counts > 1

    sample_variance = (shear_sum_squared[valid_bins] - counts[valid_bins] * mean_gamma_t[valid_bins]**2) / (counts[valid_bins] - 1)

    sample_variance = np.maximum(sample_variance, 0)

    gamma_t_error[valid_bins] = np.sqrt(sample_variance / counts[valid_bins])

    return gamma_t_error


def get_nfw_tangential_shear(halo_mass: float, source_redshift: float, radii: np.ndarray,
                             concentration: float = 4, halo_redshift: float = 0.3,
                             omega_m: float = 0.3, omega_lam: float = 0.7) -> np.ndarray:
    '''Calculate the unreduced NFW tangential shear profile.
    Args
        halo_mass (float): M_200c halo mass in solar masses / h.
        source_redshift (float): Source galaxy redshift.
        radii (ndarray): Angular separations in arcseconds.
        concentration (float): NFW concentration parameter.
        halo_redshift (float): Halo redshift.
        omega_m (float): Matter density fraction.
        omega_lam (float): Dark energy density fraction.
    Returns
        ndarray: Dimensionless, unreduced tangential shear at the input radii.
    '''
    nfw = galsim.NFWHalo(
        mass=halo_mass,
        conc=concentration,
        redshift=halo_redshift,
        omega_m=omega_m,
        omega_lam=omega_lam,
    )

    # Along the positive x-axis, gamma_t = -gamma_1.
    gamma_1, _ = nfw.getShear(
        (radii, np.zeros_like(radii)), source_redshift, reduced=False,
    )

    return -gamma_1


def get_chi2(gt_data: np.ndarray, gt_model: np.ndarray, gt_err: np.ndarray) -> float:
    '''Calculate chi-squared between the measured and model shear profiles.
    Args
        gt_data (ndarray): Measured tangential shear profile.
        gt_model (ndarray): Model tangential shear profile.
        gt_err (ndarray): Uncertainty on the measured tangential shear profile.
    Returns
        float: Chi-squared over finite bins with positive uncertainty. If no
               bins satisfy the mask, the sum is zero.
    '''
    valid = np.isfinite(gt_data) & np.isfinite(gt_model) & np.isfinite(gt_err) & (gt_err > 0)

    chi2 = np.sum(((gt_data[valid] - gt_model[valid]) / gt_err[valid])**2)

    return chi2


def get_log_likelihood(gt_data: np.ndarray, gt_model: np.ndarray, gt_err: np.ndarray) -> float:
    '''Calculate the Gaussian log-likelihood.
    Args
        gt_data (ndarray): Measured tangential shear profile.
        gt_model (ndarray): Model tangential shear profile.
        gt_err (ndarray): Uncertainty on the measured tangential shear profile.
    Returns
        float: Log-likelihood value up to an additive constant.
    '''
    chi2 = get_chi2(gt_data, gt_model, gt_err)

    return -0.5 * chi2


def get_joint_log_likelihood_grid(halo_masses: np.ndarray, source_redshifts: np.ndarray,
                                  radii: np.ndarray, gt_data: np.ndarray,
                                  gt_err: np.ndarray) -> np.ndarray:
    '''Calculate the joint log-likelihood grid over halo mass and source redshift.
    Args
        halo_masses (ndarray): M_200c grid in solar masses / h.
        source_redshifts (ndarray): Grid of source redshifts.
        radii (ndarray): Angular separations in arcseconds.
        gt_data (ndarray): Measured tangential shear profile.
        gt_err (ndarray): Uncertainty on the measured tangential shear profile.
    Returns
        ndarray: Log-likelihood grid with shape
                 (len(source_redshifts), len(halo_masses)).
    '''
    log_likelihood = np.full((len(source_redshifts), len(halo_masses)), np.nan)

    for i, source_redshift in enumerate(source_redshifts):
        for j, halo_mass in enumerate(halo_masses):
            gt_model = get_nfw_tangential_shear(
                halo_mass=halo_mass,
                source_redshift=source_redshift,
                radii=radii,
            )

            log_likelihood[i, j] = get_log_likelihood(
                gt_data,
                gt_model,
                gt_err,
            )

    return log_likelihood


def get_normalised_posterior(log_likelihood: np.ndarray) -> np.ndarray:
    '''Normalise posterior cell probabilities on an equally spaced grid.
    Args
        log_likelihood (ndarray): Log-likelihood grid.
    Returns
        ndarray: Posterior cell probabilities summing to one under a flat
                 prior over an equally spaced parameter grid.
    '''
    posterior = np.exp(log_likelihood - np.nanmax(log_likelihood))
    posterior = posterior / np.nansum(posterior)

    return posterior


def get_credible_levels(posterior: np.ndarray,
                        credible_fractions: tuple[float, ...] = (0.68, 0.95)) -> list[float]:
    '''Find posterior cell-probability levels enclosing credible fractions.
    Args
        posterior (ndarray): Finite, normalised posterior cell probabilities.
        credible_fractions (tuple): Credible fractions to enclose.
    Returns
        list: Cell-probability thresholds for the requested credible fractions.
    '''
    posterior_sorted = np.sort(posterior.ravel())[::-1]
    cumulative_probability = np.cumsum(posterior_sorted)
    cumulative_probability = cumulative_probability / cumulative_probability[-1]

    levels = []
    for fraction in credible_fractions:
        index = np.searchsorted(cumulative_probability, fraction)
        levels.append(posterior_sorted[index])

    return levels


def get_best_fit_from_posterior(halo_masses: np.ndarray, source_redshifts: np.ndarray,
                                posterior: np.ndarray) -> tuple[float, float]:
    '''Find the maximum posterior point on a two-dimensional grid.
    Args
        halo_masses (ndarray): M_200c grid in solar masses / h.
        source_redshifts (ndarray): Grid of source redshifts.
        posterior (ndarray): Posterior grid.
    Returns
        tuple: best_halo_mass, best_source_redshift
    '''
    best_index = np.unravel_index(np.nanargmax(posterior), posterior.shape)
    best_source_redshift = source_redshifts[best_index[0]]
    best_halo_mass = halo_masses[best_index[1]]

    return best_halo_mass, best_source_redshift


def get_mass_log_likelihood(halo_masses: np.ndarray, source_redshift: float,
                            radii: np.ndarray, gt_data: np.ndarray, gt_err: np.ndarray) -> np.ndarray:
    '''Calculate log-likelihood over halo mass for a fixed source redshift.
    Args
        halo_masses (ndarray): M_200c grid in solar masses / h.
        source_redshift (float): Fixed source redshift.
        radii (ndarray): Angular separations in arcseconds.
        gt_data (ndarray): Measured tangential shear profile.
        gt_err (ndarray): Uncertainty on the measured tangential shear profile.
    Returns
        ndarray: Log-likelihood values for each halo mass.
    '''
    log_likelihood = np.full(len(halo_masses), np.nan)

    for i, halo_mass in enumerate(halo_masses):
        gt_model = get_nfw_tangential_shear(halo_mass=halo_mass, source_redshift=source_redshift, radii=radii)

        log_likelihood[i] = get_log_likelihood(gt_data, gt_model, gt_err)

    return log_likelihood


def get_normalised_1d_posterior(log_likelihood: np.ndarray, parameter_grid: np.ndarray) -> np.ndarray:
    '''Convert one-dimensional log-likelihood values into a posterior density.
    Args
        log_likelihood (ndarray): Log-likelihood values.
        parameter_grid (ndarray): Strictly increasing parameter grid.
    Returns
        ndarray: Normalised posterior density.
    '''
    posterior = np.exp(log_likelihood - np.nanmax(log_likelihood))
    posterior = posterior / np.trapezoid(posterior, parameter_grid)

    return posterior
