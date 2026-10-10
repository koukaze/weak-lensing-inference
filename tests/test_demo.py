'''Check the image injection used by the synthetic walkthrough.'''

import numpy as np

from weak_lensing.analysis import get_ellip_hsm, get_nfw_tangential_shear
from weak_lensing.demo import make_synthetic_catalogue


def test_synthetic_images_recover_injected_tangential_signal():
    '''Check the tangential sign and shape convention with negligible pixel noise.'''
    images, positions, halos, truth = make_synthetic_catalogue(
        sources_per_bin=2, noise_sigma=1.0e-6
    )
    shape = get_ellip_hsm(images)
    radius = np.sqrt(np.sum(positions**2, axis=1))
    angle = np.arctan2(positions[:, 1], positions[:, 0])
    measured = -(shape[:, 0] * np.cos(2 * angle) + shape[:, 1] * np.sin(2 * angle))
    expected = get_nfw_tangential_shear(
        truth["halo_mass_msun_over_h"], truth["source_redshift"], radius
    )
    np.testing.assert_allclose(measured, expected, atol=1.0e-7, rtol=1.0e-5)
    np.testing.assert_array_equal(halos, np.zeros((1, 2)))


def test_synthetic_catalogue_repeats_from_seed():
    '''Check repeatability of both source locations and noisy image pixels.'''
    first = make_synthetic_catalogue(sources_per_bin=2, seed=42)
    second = make_synthetic_catalogue(sources_per_bin=2, seed=42)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0][0].array, second[0][0].array)
    assert first[3] == second[3]
