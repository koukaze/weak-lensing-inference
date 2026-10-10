'''Plot galaxy-shape comparisons, tangential profiles, and NFW posteriors.'''

from __future__ import annotations

from collections.abc import Sequence

import galsim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def galaxy_stamp(
    image: np.ndarray,
    title: str = 'Stamp 0',
    cmap: str = 'inferno',
) -> tuple[Figure, Axes]:
    '''Plot a single galaxy stamp.
    Args
        image (ndarray): Image array.
        title (str): Plot title.
        cmap (str): Matplotlib colour map.
    Returns
        tuple: fig, ax.
    '''
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(image, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    return fig, ax


def galaxy_stamp_grid(
    galaxy_ims: Sequence[galsim.Image],
    n_stamps: int = 16,
    n_cols: int = 4,
    cmap: str = 'inferno',
) -> tuple[Figure, np.ndarray]:
    '''Plot a grid of galaxy stamps.
    Args
        galaxy_ims (Sequence): GalSim Image objects.
        n_stamps (int): Number of stamps to show.
        n_cols (int): Number of columns.
        cmap (str): Matplotlib colour map.
    Returns
        tuple: fig, axes.
    '''
    n_rows = int(np.ceil(n_stamps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8),
                             constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for i, ax in enumerate(axes):
        if i < n_stamps:
            im = galaxy_ims[i].array
            ax.imshow(im, origin="lower", cmap=cmap)
            ax.set_title(f"Stamp {i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axes


def invalid_stamp_grid(
    galaxy_ims: Sequence[galsim.Image],
    invalid_indices: np.ndarray,
    n_stamps: int = 16,
    n_cols: int = 4,
    cmap: str = 'inferno',
) -> tuple[Figure, np.ndarray]:
    '''Plot stamps with invalid shape measurements.
    Args
        galaxy_ims (Sequence): GalSim Image objects.
        invalid_indices (ndarray): Indices of invalid measurements.
        n_stamps (int): Number of stamps to show.
        n_cols (int): Number of columns.
        cmap (str): Matplotlib colour map.
    Returns
        tuple: fig, axes.
    '''
    n_rows = int(np.ceil(n_stamps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8),
                             constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for ax, idx in zip(axes, invalid_indices[:n_stamps]):
        im = galaxy_ims[idx].array
        ax.imshow(im, origin="lower", cmap=cmap)
        ax.set_title(f"Stamp {idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(invalid_indices[:n_stamps]):]:
        ax.set_axis_off()

    return fig, axes


def shape_component_distributions(
    e1_unweighted: np.ndarray,
    e2_unweighted: np.ndarray,
    e1_weighted: np.ndarray,
    e2_weighted: np.ndarray,
    g1_hsm: np.ndarray,
    g2_hsm: np.ndarray,
    valid_all: np.ndarray,
) -> tuple[Figure, np.ndarray]:
    '''Plot distributions of the two shape components.
    Args
        e1_unweighted (ndarray): Unweighted first component.
        e2_unweighted (ndarray): Unweighted second component.
        e1_weighted (ndarray): Weighted first component.
        e2_weighted (ndarray): Weighted second component.
        g1_hsm (ndarray): HSM first component.
        g2_hsm (ndarray): HSM second component.
        valid_all (ndarray): Mask selecting stamps valid for all methods.
    Returns
        tuple: fig, axes.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    bins = np.linspace(-1, 1, 100)

    axes[0].hist(e1_unweighted[valid_all], bins=bins, alpha=0.5,
                 density=True, label="Unweighted")
    axes[0].hist(e1_weighted[valid_all], bins=bins, alpha=0.5,
                 density=True, label="Weighted")
    axes[0].hist(g1_hsm[valid_all], bins=bins, alpha=0.5,
                 density=True, label="HSM")
    axes[0].set_xlabel(r"$e_1$ or $g_1$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"Component 1 distribution")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].hist(e2_unweighted[valid_all], bins=bins, alpha=0.5,
                 density=True, label="Unweighted")
    axes[1].hist(e2_weighted[valid_all], bins=bins, alpha=0.5,
                 density=True, label="Weighted")
    axes[1].hist(g2_hsm[valid_all], bins=bins, alpha=0.5,
                 density=True, label="HSM")
    axes[1].set_xlabel(r"$e_2$ or $g_2$")
    axes[1].set_ylabel("Density")
    axes[1].set_title(r"Component 2 distribution")
    axes[1].set_yscale("log")
    axes[1].legend()

    return fig, axes


def shape_scatter_vs_hsm(
    e1_unweighted: np.ndarray,
    e2_unweighted: np.ndarray,
    e1_weighted: np.ndarray,
    e2_weighted: np.ndarray,
    g1_hsm: np.ndarray,
    g2_hsm: np.ndarray,
    valid_all: np.ndarray,
) -> tuple[Figure, np.ndarray]:
    '''Plot moment-method shape estimates against HSM estimates.
    Args
        e1_unweighted (ndarray): Unweighted first component.
        e2_unweighted (ndarray): Unweighted second component.
        e1_weighted (ndarray): Weighted first component.
        e2_weighted (ndarray): Weighted second component.
        g1_hsm (ndarray): HSM first component.
        g2_hsm (ndarray): HSM second component.
        valid_all (ndarray): Mask selecting stamps valid for all methods.
    Returns
        tuple: fig, axes.
    '''
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    axes[0, 0].scatter(g1_hsm[valid_all], e1_unweighted[valid_all],
                       s=2, alpha=0.2)
    axes[0, 0].set_xlabel(r"HSM $g_1$")
    axes[0, 0].set_ylabel(r"Unweighted $e_1$")
    axes[0, 0].set_title(r"Unweighted $e_1$ vs HSM $g_1$")

    axes[0, 1].scatter(g2_hsm[valid_all], e2_unweighted[valid_all],
                       s=2, alpha=0.2)
    axes[0, 1].set_xlabel(r"HSM $g_2$")
    axes[0, 1].set_ylabel(r"Unweighted $e_2$")
    axes[0, 1].set_title(r"Unweighted $e_2$ vs HSM $g_2$")

    axes[1, 0].scatter(g1_hsm[valid_all], e1_weighted[valid_all],
                       s=2, alpha=0.2)
    axes[1, 0].set_xlabel(r"HSM $g_1$")
    axes[1, 0].set_ylabel(r"Weighted $e_1$")
    axes[1, 0].set_title(r"Weighted $e_1$ vs HSM $g_1$")

    axes[1, 1].scatter(g2_hsm[valid_all], e2_weighted[valid_all],
                       s=2, alpha=0.2)
    axes[1, 1].set_xlabel(r"HSM $g_2$")
    axes[1, 1].set_ylabel(r"Weighted $e_2$")
    axes[1, 1].set_title(r"Weighted $e_2$ vs HSM $g_2$")

    return fig, axes


def tangential_shear_profiles(
    profiles: Sequence[tuple],
) -> tuple[Figure, Axes]:
    '''Plot mean tangential shear profiles.
    Args
        profiles (Sequence): Tuples of radius, mean shear, and label.
    Returns
        tuple: fig, ax.
    '''
    fig, ax = plt.subplots(figsize=(6, 4))

    for radius, mean_shear, label in profiles:
        ax.plot(radius, mean_shear, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel(r"Mean tangential shear $\gamma_t$")
    ax.legend()

    return fig, ax


def mean_shear_with_errorbars(
    profiles: Sequence[tuple],
) -> tuple[Figure, Axes]:
    '''Plot mean tangential shear profiles with error bars.
    Args
        profiles (Sequence): List of tuples containing radius, mean shear,
                         shear uncertainty, and label.
    Returns
        tuple: fig, ax.
    '''
    fig, ax = plt.subplots(figsize=(6, 4))

    for radius, mean_shear, shear_error, label in profiles:
        ax.errorbar(
            radius,
            mean_shear,
            yerr=shear_error,
            marker="o",
            capsize=3,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel(r"Mean tangential shear $\gamma_t$")
    ax.legend()

    return fig, ax


def nfw_comparison(
    radius: np.ndarray,
    gt_data: np.ndarray,
    gt_error: np.ndarray,
    gt_model: np.ndarray,
) -> tuple[Figure, Axes]:
    '''Plot measured tangential shear against an NFW theory profile.
    Args
        radius (ndarray): Radial bin centres.
        gt_data (ndarray): Measured tangential shear.
        gt_error (ndarray): Measurement uncertainties.
        gt_model (ndarray): NFW prediction.
    Returns
        tuple: fig, ax.
    '''
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(radius, gt_data, yerr=gt_error, marker="o",
                label="Measured HSM")
    ax.plot(radius, gt_model, label="NFW theory")
    ax.set_xscale("log")
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel(r"$\gamma_t$")
    ax.legend()

    return fig, ax


def joint_posterior(
    halo_masses: np.ndarray,
    source_redshifts: np.ndarray,
    posterior: np.ndarray,
    level_68: float,
    level_95: float,
    best_halo_mass: float,
    best_source_redshift: float,
) -> tuple[Figure, Axes]:
    '''Plot the joint posterior over halo mass and source redshift.
    Args
        halo_masses (ndarray): M_200c grid in solar masses / h.
        source_redshifts (ndarray): Source redshift grid.
        posterior (ndarray): Normalised posterior cell probabilities.
        level_68 (float): 68% credible cell-probability threshold.
        level_95 (float): 95% credible cell-probability threshold.
        best_halo_mass (float): Maximum-posterior M_200c in solar masses / h.
        best_source_redshift (float): Maximum-posterior source redshift.
    Returns
        tuple: fig, ax.
    '''
    mass_grid, redshift_grid = np.meshgrid(halo_masses, source_redshifts)

    fig, ax = plt.subplots(figsize=(6, 5))
    contourf = ax.contourf(
        mass_grid,
        redshift_grid,
        posterior,
        levels=40,
        cmap="magma",
    )
    # Coincident HPD thresholds can occur on coarse or concentrated grids.
    levels = np.unique([level_95, level_68])
    levels = levels[(levels > np.nanmin(posterior)) & (levels < np.nanmax(posterior))]
    if len(levels):
        ax.contour(
            mass_grid,
            redshift_grid,
            posterior,
            levels=levels,
            colors="white",
            linewidths=1.2,
        )
    ax.plot(
        best_halo_mass,
        best_source_redshift,
        marker="x",
        color="cyan",
        markersize=8,
        label="Maximum posterior",
    )

    fig.colorbar(contourf, ax=ax, label="Posterior cell probability")
    ax.set_xlabel(r"Halo mass $M_{200c}$ [$M_\odot/h$]")
    ax.set_ylabel(r"Source redshift $z_\mathrm{s}$")
    ax.legend()

    return fig, ax


def mass_posterior(
    halo_masses: np.ndarray,
    mass_posterior_fixed_z: np.ndarray,
    best_mass_fixed_z: float,
    fixed_source_redshift: float,
) -> tuple[Figure, Axes]:
    '''Plot the one-dimensional halo mass posterior.
    Args
        halo_masses (ndarray): M_200c grid in solar masses / h.
        mass_posterior_fixed_z (ndarray): Posterior density.
        best_mass_fixed_z (float): Maximum-posterior M_200c in solar masses / h.
        fixed_source_redshift (float): Fixed source redshift.
    Returns
        tuple: fig, ax.
    '''
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(halo_masses, mass_posterior_fixed_z)
    ax.axvline(best_mass_fixed_z, color="grey", linestyle="--",
               label="Maximum posterior")
    ax.set_xlabel(r"Halo mass $M_{200c}$ [$M_\odot/h$]")
    ax.set_ylabel(r"Posterior density [$(M_\odot/h)^{-1}$]")
    ax.set_title(
        rf"Halo mass posterior for $z_\mathrm{{s}} = {fixed_source_redshift}$"
    )
    ax.legend()

    return fig, ax
