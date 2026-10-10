'''Generate a small synthetic image catalogue for the weak-lensing example.'''

import galsim
import numpy as np

from .analysis import get_nfw_tangential_shear


def make_synthetic_catalogue(
    sources_per_bin: int = 200,
    halo_mass: float = 5.0e14,
    source_redshift: float = 0.6,
    noise_sigma: float = 20.0,
    seed: int = 625,
) -> tuple:
    '''Generate noisy Gaussian stamps with an injected tangential NFW signal.

    Args
        sources_per_bin (int): Number of galaxies at each of ten radial bin centres.
        halo_mass (float): Injected halo mass in solar masses divided by h.
        source_redshift (float): Redshift of all source galaxies; greater than 0.3.
        noise_sigma (float): Standard deviation of additive Gaussian pixel noise.
        seed (int): Seed for source angles and pixel noise.

    Returns
        tuple: Galaxy images, source positions of shape (n_sources, 2), one halo
            position of shape (1, 2), and a dictionary of simulation parameters.
    '''
    if sources_per_bin < 2:
        raise ValueError("sources_per_bin must be at least 2.")
    if halo_mass <= 0 or source_redshift <= 0.3 or noise_sigma <= 0:
        raise ValueError("Use positive mass and noise, and source_redshift > 0.3.")

    rng = np.random.default_rng(seed)
    edges = np.logspace(np.log10(100.0), np.log10(3600.0), 11)
    radii = np.sqrt(edges[:-1] * edges[1:])
    source_radii = np.repeat(radii, sources_per_bin)
    phi = rng.uniform(0.0, 2.0 * np.pi, len(source_radii))
    source_positions = np.column_stack(
        (source_radii * np.cos(phi), source_radii * np.sin(phi))
    )
    halo_positions = np.zeros((1, 2))
    gamma_t = get_nfw_tangential_shear(halo_mass, source_redshift, radii)
    gamma_t = np.repeat(gamma_t, sources_per_bin)
    g1 = -gamma_t * np.cos(2.0 * phi)
    g2 = -gamma_t * np.sin(2.0 * phi)

    # Inject gamma as the image shape parameter using the weak-shear approximation.
    # The example excludes convergence, intrinsic shapes, PSF convolution and blends.
    galaxy = galsim.Gaussian(sigma=2.2, flux=10000.0)
    images = []
    for component_1, component_2 in zip(g1, g2):
        image = galaxy.shear(g1=component_1, g2=component_2).drawImage(
            nx=50, ny=50, scale=1.0, method="no_pixel"
        )
        pixels = image.array + rng.normal(0.0, noise_sigma, image.array.shape)
        images.append(galsim.ImageF(pixels.astype(np.float32), scale=1.0))

    truth = {
        "halo_mass_msun_over_h": halo_mass,
        "source_redshift": source_redshift,
        "halo_redshift": 0.3,
        "concentration": 4,
        "omega_m": 0.3,
        "omega_lam": 0.7,
        "sources_per_bin": sources_per_bin,
        "n_sources": len(images),
        "noise_sigma": noise_sigma,
        "seed": seed,
        "stamp_size_pixels": 50,
        "gaussian_sigma_pixels": 2.2,
        "flux": 10000.0,
        "r_min_arcsec": 100.0,
        "r_max_arcsec": 3600.0,
        "n_bins": 10,
    }
    return images, source_positions, halo_positions, truth


def main() -> None:
    '''Run the synthetic image analysis and save its parameters and results.'''
    import argparse
    import json
    from pathlib import Path

    from .pipeline import analyse_shapes, measure_images, save_analysis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--seed", type=int, default=625)
    parser.add_argument("--sources-per-bin", type=int, default=200)
    parser.add_argument("--noise-sigma", type=float, default=20.0)
    parser.add_argument("--grid-size", type=int, default=120)
    args = parser.parse_args()

    images, sources, halos, truth = make_synthetic_catalogue(
        sources_per_bin=args.sources_per_bin,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    shapes = measure_images(images, len(images))
    results = analyse_shapes(shapes, sources, halos, grid_size=args.grid_size)
    results["summary"]["input_kind"] = "Synthetic Gaussian-image demonstration"
    results["summary"]["source_subset"] = False
    save_analysis(results, args.output_dir)
    (args.output_dir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
