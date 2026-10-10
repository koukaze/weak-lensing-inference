'''Check shape conventions, pair statistics, and NFW posterior calculations.'''

import galsim
import numpy as np
import pytest

from weak_lensing import analysis, plotting


def gaussian_image(sigma_x: float, sigma_y: float, angle: float = 0.0) -> galsim.Image:
    '''Return a well-sampled Gaussian image with a known covariance.
    Args
        sigma_x (float): Gaussian width along the first principal axis, in pixels.
        sigma_y (float): Gaussian width along the second principal axis, in pixels.
        angle (float): Position angle of the first axis, in radians.
    Returns
        Image: Centred 81 by 81 pixel Gaussian stamp.
    '''
    y, x = np.indices((81, 81), dtype=float)
    x -= 40
    y -= 40
    major = x * np.cos(angle) + y * np.sin(angle)
    minor = -x * np.sin(angle) + y * np.cos(angle)
    pixels = np.exp(-0.5 * ((major / sigma_x)**2 + (minor / sigma_y)**2))
    return galsim.Image(pixels, scale=1.0)


def test_unweighted_circular_and_elliptical_gaussians():
    images = [gaussian_image(3, 3), gaussian_image(4, 2)]
    result = analysis.get_ellip_unweighted(images)
    np.testing.assert_allclose(result, [[0, 0], [1 / 3, 0]], atol=1e-12)


def test_second_component_retains_coursework_normalisation():
    result = analysis.get_ellip_unweighted([gaussian_image(4, 2, np.pi / 4)])
    np.testing.assert_allclose(result, [[0, 1 / 6]], atol=1e-12)


def test_gaussian_weight_matches_analytic_product_covariance():
    sigma_w = 4.0 / (2 * np.sqrt(2 * np.log(2)))
    a = 1 / (1 / 16 + 1 / sigma_w**2)
    b = 1 / (1 / 4 + 1 / sigma_w**2)
    expected_e1 = (a - b) / (a + b + 2 * np.sqrt(a * b))
    result = analysis.get_ellip_weighted([gaussian_image(4, 2)])
    np.testing.assert_allclose(result, [[expected_e1, 0]], atol=1e-7)


def test_shapes_are_invariant_to_flux_rescaling():
    image = gaussian_image(4, 2, 0.3)
    brighter = galsim.Image(image.array * 73, scale=1.0)
    for measure in (analysis.get_ellip_unweighted, analysis.get_ellip_weighted):
        np.testing.assert_allclose(measure([image]), measure([brighter]), atol=1e-13)


def test_zero_flux_images_produce_nan_shapes():
    image = galsim.Image(np.zeros((9, 9)), scale=1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        assert np.isnan(analysis.get_ellip_unweighted([image])).all()
        assert np.isnan(analysis.get_ellip_weighted([image])).all()
    assert np.isnan(analysis.get_ellip_hsm([image])).all()


def test_hsm_returns_observed_reduced_shape_components():
    images = [gaussian_image(4, 2), gaussian_image(4, 2, np.pi / 4)]
    result = analysis.get_ellip_hsm(images)
    np.testing.assert_allclose(result, [[1 / 3, 0], [0, 1 / 3]], atol=1e-7)


def test_fits_round_trip_preserves_stamp_order(tmp_path):
    images = [gaussian_image(3, 3), gaussian_image(4, 2)]
    filename = tmp_path / 'stamps.fits'
    galsim.fits.writeMulti(images, str(filename))
    loaded = analysis.load_galaxy_stamps(str(filename))
    assert len(loaded) == 2
    for original, actual in zip(images, loaded):
        np.testing.assert_array_equal(original.array, actual.array)


def test_tangential_projection_on_axes_and_diagonal():
    sources = np.array([[2, 0], [0, 2], [np.sqrt(2), np.sqrt(2)]])
    g1 = np.array([-0.1, 0.2, 0.0])
    g2 = np.array([0.0, 0.0, -0.3])
    radius, mean, counts, sum_squared = analysis.get_tangential_shear_profile(
        g1, g2, sources, np.array([[0, 0]]), n_bins=1, r_min=1, r_max=4,
    )
    np.testing.assert_allclose(radius, [2])
    np.testing.assert_allclose(mean, [0.2])
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_allclose(sum_squared, [0.01 + 0.04 + 0.09])


def test_logarithmic_bin_boundaries_and_invalid_shapes():
    distances = np.array([1, 9.999, 10, 99.999, 100, 0.999, 3])
    sources = np.column_stack((distances, np.zeros_like(distances)))
    g1 = np.array([-0.1, -0.3, -0.2, -0.4, -9, -9, np.nan])
    radius, mean, counts, squared = analysis.get_tangential_shear_profile(
        g1, np.zeros_like(g1), sources, np.array([[0, 0]]),
        n_bins=2, r_min=1, r_max=100,
    )
    np.testing.assert_allclose(radius, np.sqrt([10, 1000]))
    np.testing.assert_array_equal(counts, [2, 2])
    np.testing.assert_allclose(mean, [0.2, 0.3])
    np.testing.assert_allclose(squared, [0.1, 0.2])


def test_stack_counts_each_halo_source_pair():
    sources = np.array([[2, 0], [3, 0]])
    one = analysis.get_tangential_shear_profile(
        np.array([-0.1, -0.3]), np.zeros(2), sources, np.array([[0, 0]]),
        n_bins=2, r_min=1, r_max=100,
    )
    two = analysis.get_tangential_shear_profile(
        np.array([-0.1, -0.3]), np.zeros(2), sources,
        np.array([[0, 0], [0, 0]]), n_bins=2, r_min=1, r_max=100,
    )
    np.testing.assert_allclose(one[1], two[1], equal_nan=True)
    np.testing.assert_array_equal(two[2], 2 * one[2])
    np.testing.assert_allclose(two[3], 2 * one[3])
    assert np.isnan(two[1][1])


def test_standard_errors_match_hand_calculation():
    errors = analysis.get_shear_uncertainties(
        np.array([2.0, 1.0, np.nan, 1.0]),
        np.array([14.0, 1.0, 0.0, 2.0 - 1e-15]),
        np.array([3, 1, 0, 2]),
    )
    np.testing.assert_allclose(errors[0], 1 / np.sqrt(3))
    assert np.isnan(errors[1:3]).all()
    assert errors[3] == 0.0


def test_chi_squared_masks_invalid_bins_and_nonpositive_errors():
    data = np.array([2, 3, np.nan, 100, 100, 100])
    model = np.array([1, 2, 0, 0, 0, np.nan])
    error = np.array([1, 2, 1, 0, -1, 1])
    assert analysis.get_chi2(data, model, error) == 1.25
    assert analysis.get_log_likelihood(data, model, error) == -0.625


def test_posterior_normalisation_is_stable_to_log_likelihood_offset():
    log_likelihood = np.array([[0.0, -1.0], [-2.0, -3.0]])
    posterior = analysis.get_normalised_posterior(log_likelihood)
    shifted = analysis.get_normalised_posterior(log_likelihood + 10000)
    np.testing.assert_allclose(posterior, shifted)
    assert posterior.sum() == pytest.approx(1.0)
    masses = np.array([3e14, 5e14])
    redshifts = np.array([0.4, 0.6])
    assert analysis.get_best_fit_from_posterior(masses, redshifts, posterior) == (3e14, 0.4)


def test_highest_posterior_cell_thresholds():
    posterior = np.array([[0.4, 0.3], [0.2, 0.1]])
    levels = analysis.get_credible_levels(posterior, (0.65, 0.95))
    np.testing.assert_allclose(levels, [0.3, 0.1])


def test_one_dimensional_posterior_is_density_over_mass():
    masses = np.linspace(3e14, 14.8e14, 120)
    log_likelihood = -0.5 * ((masses - 7e14) / 1e14)**2
    posterior = analysis.get_normalised_1d_posterior(log_likelihood, masses)
    assert np.trapezoid(posterior, masses) == pytest.approx(1.0)
    np.testing.assert_allclose(
        posterior,
        analysis.get_normalised_1d_posterior(log_likelihood + 1e5, masses),
        rtol=1e-10,
    )


def test_joint_and_fixed_redshift_inference_recover_grid_point():
    radii = np.geomspace(100, 3600, 10)
    masses = np.array([3e14, 6e14, 9e14])
    redshifts = np.array([0.4, 0.6, 0.8])
    measured = analysis.get_nfw_tangential_shear(6e14, 0.6, radii)
    errors = np.full_like(measured, 0.002)
    likelihood = analysis.get_joint_log_likelihood_grid(
        masses, redshifts, radii, measured, errors,
    )
    assert likelihood.shape == (3, 3)
    posterior = analysis.get_normalised_posterior(likelihood)
    assert analysis.get_best_fit_from_posterior(masses, redshifts, posterior) == (6e14, 0.6)
    fixed = analysis.get_mass_log_likelihood(masses, 0.6, radii, measured, errors)
    np.testing.assert_allclose(fixed, likelihood[1])


def test_nfw_profile_matches_unreduced_reference_values():
    radii = np.array([100, 200, 400, 800, 1600, 3200.0])
    expected = np.array([
        0.04881386707431452,
        0.03025637066378464,
        0.01539239569519777,
        0.006601481661786824,
        0.002486504353377513,
        0.0008537259988535359,
    ])
    actual = analysis.get_nfw_tangential_shear(7.462e14, 0.466, radii)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_joint_posterior_plot_handles_coincident_credible_thresholds():
    import matplotlib.pyplot as plt

    masses = np.linspace(3e14, 14.8e14, 5)
    redshifts = np.linspace(0.31, 0.8, 5)
    posterior = np.zeros((5, 5))
    posterior[2, 2] = 1.0
    level_68, level_95 = analysis.get_credible_levels(posterior)
    assert level_68 == level_95 == 1.0
    figure, axis = plotting.joint_posterior(
        masses, redshifts, posterior, level_68, level_95, masses[2], redshifts[2],
    )
    try:
        figure.canvas.draw()
        assert len(axis.collections) == 1
    finally:
        plt.close(figure)
