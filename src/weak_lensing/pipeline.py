'''Run shape measurement, profile stacking and halo inference on local inputs.'''

from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
import csv
from itertools import islice
import json
from pathlib import Path

from astropy.io import fits
import galsim
import matplotlib.pyplot as plt
import numpy as np

from . import analysis, plotting

METHODS = ('unweighted', 'weighted', 'hsm')


def iter_galaxy_images(filepath: str | Path) -> Iterator[galsim.Image]:
    '''Read two-dimensional FITS image HDUs in catalogue order.
    Args
        filepath (str or Path): Uncompressed multi-extension galaxy-stamp FITS file.
    Returns
        Iterator: GalSim images with the WCS and pixel bounds read from each header.
    '''
    with fits.open(filepath, memmap=True, lazy_load_hdus=True) as hdus:
        for hdu in hdus:
            if hdu.data is None or hdu.data.ndim != 2:
                raise ValueError('Every HDU must contain one two-dimensional galaxy stamp.')
            # A one-HDU view avoids asking GalSim to materialise the full image list.
            image = galsim.fits.read(hdu_list=fits.HDUList([hdu]), hdu=0, compression=None)
            yield image
            del hdu.data


def measure_images(
    images: Iterable[galsim.Image],
    expected_count: int,
    *,
    batch_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    '''Measure three shape estimators while retaining image order and failed rows.
    Args
        images (Iterable): Galaxy-stamp images in source-catalogue order.
        expected_count (int): Number of catalogue sources, greater than zero.
        batch_size (int): Maximum number of images processed together.
        progress (Callable or None): Optional callback receiving completed and total counts.
    Returns
        dict: Estimator names mapped to arrays of shape (expected_count, 2).
    '''
    if expected_count < 1 or batch_size < 1:
        raise ValueError('expected_count and batch_size must be positive.')
    shapes = {name: np.full((expected_count, 2), np.nan) for name in METHODS}
    iterator = iter(images)
    offset = 0
    while batch := list(islice(iterator, batch_size)):
        end = offset + len(batch)
        if end > expected_count:
            raise ValueError('More galaxy stamps than source-position rows.')
        shapes['unweighted'][offset:end] = analysis.get_ellip_unweighted(batch)
        shapes['weighted'][offset:end] = analysis.get_ellip_weighted(batch)
        shapes['hsm'][offset:end] = analysis.get_ellip_hsm(batch)
        offset = end
        if progress is not None:
            progress(offset, expected_count)
    if offset != expected_count:
        raise ValueError(f'Found {offset} stamps for {expected_count} source positions.')
    return shapes


def analyse_shapes(
    shapes: dict[str, np.ndarray],
    source_positions: np.ndarray,
    halo_positions: np.ndarray,
    *,
    grid_size: int = 120,
) -> dict:
    '''Stack shape components and evaluate the coursework posterior grids.
    Args
        shapes (dict): Unweighted, weighted and HSM arrays of shape (n_sources, 2).
        source_positions (ndarray): Source coordinates in arcseconds, shape (n_sources, 2).
        halo_positions (ndarray): Halo coordinates in arcseconds, shape (n_halos, 2).
        grid_size (int): Number of uniformly spaced samples per inference parameter.
    Returns
        dict: Profiles, posterior arrays, parameter grids and a numerical summary.
    '''
    source_positions = np.asarray(source_positions, dtype=float)
    halo_positions = np.asarray(halo_positions, dtype=float)
    for positions in (source_positions, halo_positions):
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0 or not np.isfinite(positions).all():
            raise ValueError('Position arrays must have shape (n, 2), with finite coordinates.')
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 5:
        raise ValueError('grid_size must be an integer of at least five.')
    output = {}
    invalid = {}
    for name in METHODS:
        components = np.asarray(shapes[name], dtype=float)
        if components.shape != (len(source_positions), 2):
            raise ValueError(f'{name} measurements must match the source-position row order and count.')
        radius, mean, counts, sum_squared = analysis.get_tangential_shear_profile(
            components[:, 0], components[:, 1], source_positions, halo_positions,
        )
        output['radii_arcsec'] = radius
        output[f'mean_{name}'] = mean
        output[f'counts_{name}'] = counts
        output[f'sum_squared_{name}'] = sum_squared
        output[f'error_{name}'] = analysis.get_shear_uncertainties(mean, sum_squared, counts)
        invalid[name] = int((~np.isfinite(components).all(axis=1)).sum())

    valid = np.isfinite(output['mean_hsm']) & np.isfinite(output['error_hsm']) & (output['error_hsm'] > 0)
    if not valid.any():
        raise ValueError('The HSM profile has no bins with finite data and positive uncertainty.')
    masses = np.linspace(3.0e14, 14.8e14, grid_size)
    redshifts = np.linspace(0.31, 0.8, grid_size)
    log_grid = analysis.get_joint_log_likelihood_grid(
        masses, redshifts, radius, output['mean_hsm'], output['error_hsm'],
    )
    posterior = analysis.get_normalised_posterior(log_grid)
    best_mass, best_redshift = analysis.get_best_fit_from_posterior(masses, redshifts, posterior)
    fixed_log = analysis.get_mass_log_likelihood(masses, 0.6, radius, output['mean_hsm'], output['error_hsm'])
    fixed_posterior = analysis.get_normalised_1d_posterior(fixed_log, masses)
    output.update({
        'halo_masses_msun_over_h': masses, 'source_redshifts': redshifts,
        'joint_log_likelihood': log_grid, 'joint_posterior': posterior,
        'fixed_z_mass_posterior': fixed_posterior,
        'summary': {
            'n_sources': len(source_positions), 'n_halos': len(halo_positions),
            'invalid_shapes': invalid, 'valid_hsm_inference_bins': int(valid.sum()),
            'joint_best_mass_msun_over_h': float(best_mass),
            'joint_best_source_redshift': float(best_redshift),
            'fixed_source_redshift': 0.6,
            'fixed_z_best_mass_msun_over_h': float(masses[np.nanargmax(fixed_posterior)]),
            'grid_size': grid_size, 'mass_range_msun_over_h': [3.0e14, 14.8e14],
            'source_redshift_range': [0.31, 0.8],
            'n_bins': 10, 'radius_range_arcsec': [100.0, 3600.0],
            'concentration': 4, 'halo_redshift': 0.3, 'omega_m': 0.3, 'omega_lam': 0.7,
        },
    })
    return output


def save_analysis(results: dict, output_dir: str | Path) -> None:
    '''Save numeric results and compact profile and posterior figures.
    Args
        results (dict): Output from analyse_shapes.
        output_dir (str or Path): Destination for generated NPZ, CSV, JSON and figures.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = results['summary']
    np.savez_compressed(output_dir / 'analysis_results.npz', **{k: v for k, v in results.items() if k != 'summary'})
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output_dir / 'profiles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'radius_arcsec', 'mean_tangential_shape', 'standard_error', 'pair_count'])
        for name in METHODS:
            writer.writerows(zip([name] * len(results['radii_arcsec']), results['radii_arcsec'],
                                 results[f'mean_{name}'], results[f'error_{name}'], results[f'counts_{name}']))
    profiles = [(results['radii_arcsec'], results[f'mean_{name}'], results[f'error_{name}'], name.upper() if name == 'hsm' else name.capitalize()) for name in METHODS]
    levels = analysis.get_credible_levels(results['joint_posterior'])
    figures = {
        'shear_profiles': plotting.mean_shear_with_errorbars(profiles)[0],
        'joint_posterior': plotting.joint_posterior(
            results['halo_masses_msun_over_h'], results['source_redshifts'], results['joint_posterior'],
            *levels, summary['joint_best_mass_msun_over_h'], summary['joint_best_source_redshift'],
        )[0],
        'fixed_redshift_mass': plotting.mass_posterior(
            results['halo_masses_msun_over_h'], results['fixed_z_mass_posterior'],
            summary['fixed_z_best_mass_msun_over_h'], 0.6,
        )[0],
    }
    for name, figure in figures.items():
        figure.savefig(output_dir / f'{name}.png', dpi=180, bbox_inches='tight')
        figure.savefig(output_dir / f'{name}.svg', bbox_inches='tight')
        plt.close(figure)


def run_coursework(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    grid_size: int = 120,
    max_sources: int | None = None,
    batch_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    '''Run the analysis using external coursework files without copying the inputs.
    Args
        data_dir (str or Path): Directory containing the three specified coursework files.
        output_dir (str or Path): Destination for derived results.
        grid_size (int): Number of grid samples for each inference parameter.
        max_sources (int or None): Optional leading-source subset for software checks.
        batch_size (int): Number of galaxy images measured together.
        progress (Callable or None): Optional completed/total measurement callback.
    Returns
        dict: Numerical summary, including the number of input sources analysed.
    '''
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    source_positions = np.loadtxt(data_dir / 'source_positions.txt', ndmin=2)
    halo_positions = np.loadtxt(data_dir / 'halo_positions.txt', ndmin=2)
    if max_sources is not None:
        if isinstance(max_sources, bool) or not isinstance(max_sources, int) or max_sources < 1:
            raise ValueError('max_sources must be a positive integer or None.')
        source_positions = source_positions[:max_sources]
    with closing(iter_galaxy_images(data_dir / 'galaxy_stamps.fits')) as images:
        selected = images if max_sources is None else islice(images, len(source_positions))
        shapes = measure_images(selected, len(source_positions), batch_size=batch_size, progress=progress)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'shape_measurements.npz', **shapes)
    results = analyse_shapes(shapes, source_positions, halo_positions, grid_size=grid_size)
    results['summary']['input_kind'] = 'Supplied coursework simulations'
    results['summary']['source_subset'] = max_sources is not None
    save_analysis(results, output_dir)
    return results['summary']
