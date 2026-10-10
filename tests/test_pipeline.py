'''Check catalogue alignment, batched measurements and exported inference products.'''

from itertools import islice

import galsim
import numpy as np
import pytest

from weak_lensing import analysis
from weak_lensing.demo import make_synthetic_catalogue
from weak_lensing.pipeline import (
    METHODS, analyse_shapes, iter_galaxy_images, measure_images, save_analysis,
)


def test_fits_reader_preserves_pixels_wcs_and_batch_order(tmp_path):
    '''Streaming reads and batch boundaries must preserve the GalSim measurements.'''
    images = [galsim.Gaussian(sigma=2.0, flux=1000).shear(g1=g, g2=0.02).drawImage(
        nx=20, ny=20, scale=0.4) for g in (0.01, 0.07, -0.1, 0.13)]
    path = tmp_path / 'stamps.fits'
    galsim.fits.writeMulti(images, str(path))
    expected = galsim.fits.readMulti(str(path))
    actual = list(iter_galaxy_images(path))
    for loaded, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(loaded.array, reference.array)
        assert loaded.bounds == reference.bounds
        assert loaded.wcs == reference.wcs
    batches = measure_images(actual, len(actual), batch_size=3)
    for method, function in zip(METHODS, (analysis.get_ellip_unweighted,
                                        analysis.get_ellip_weighted, analysis.get_ellip_hsm)):
        np.testing.assert_array_equal(batches[method], function(expected))


@pytest.mark.parametrize('expected_count', [3, 5])
def test_measurement_rejects_catalogue_count_mismatch(expected_count):
    '''A missing or extra stamp must not silently shift catalogue alignment.'''
    image = galsim.Gaussian(sigma=2).drawImage(nx=16, ny=16, scale=1)
    with pytest.raises(ValueError, match='stamps|positions'):
        measure_images([image] * 4, expected_count, batch_size=2)


def test_measurement_preserves_failed_rows():
    '''Failed shape measurements keep their source row and method-specific mask.'''
    image = galsim.Gaussian(sigma=2, flux=1000).drawImage(nx=16, ny=16, scale=1)
    blank = galsim.ImageF(16, 16, scale=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        shapes = measure_images([image, blank, image], 3, batch_size=2)
    for values in shapes.values():
        assert np.isfinite(values[[0, 2]]).all()
        assert np.isnan(values[1]).all()


def test_inference_rejects_profile_without_information():
    '''An empty valid-bin mask must not produce an apparently informative posterior.'''
    positions = np.array([[200.0, 0.0], [400.0, 0.0]])
    shapes = {name: np.full((2, 2), np.nan) for name in METHODS}
    with pytest.raises(ValueError, match='no bins'):
        analyse_shapes(shapes, positions, np.zeros((1, 2)), grid_size=5)


def test_pipeline_exports_normalised_results(tmp_path):
    '''A synthetic catalogue produces finite, normalized inference and readable outputs.'''
    images, sources, halos, _ = make_synthetic_catalogue(sources_per_bin=20)
    shapes = measure_images(images, len(images), batch_size=37)
    result = analyse_shapes(shapes, sources, halos, grid_size=15)
    assert result['summary']['n_sources'] == 200
    assert result['summary']['n_halos'] == 1
    assert result['counts_hsm'].sum() == 200
    np.testing.assert_allclose(result['joint_posterior'].sum(), 1)
    np.testing.assert_allclose(np.trapezoid(result['fixed_z_mass_posterior'],
                                          result['halo_masses_msun_over_h']), 1)
    save_analysis(result, tmp_path)
    with np.load(tmp_path / 'analysis_results.npz') as saved:
        np.testing.assert_array_equal(saved['mean_hsm'], result['mean_hsm'])
        np.testing.assert_array_equal(saved['joint_posterior'], result['joint_posterior'])
    assert len((tmp_path / 'profiles.csv').read_text().splitlines()) == 31
    for stem in ('shear_profiles', 'joint_posterior', 'fixed_redshift_mass'):
        for extension in ('png', 'svg'):
            assert (tmp_path / f'{stem}.{extension}').stat().st_size > 1000
