# Data format and outputs

## External coursework inputs

The supplied data are not distributed in this repository. The coursework instructions require data and large generated files to remain outside the repository. Use a local copy through `LENSING_DATA_DIR` or `--data-dir`.

| File | Shape and convention |
| --- | --- |
| `galaxy_stamps.fits` | 100,000 image HDUs, each a 50 × 50 float32 stamp; the primary HDU also contains an image. GalSim pixel scale is 1. |
| `source_positions.txt` | 100,000 rows × 2 columns; flat Cartesian source coordinates `(x, y)` in arcseconds. |
| `halo_positions.txt` | 100 rows × 2 columns; halo coordinates in the same frame and units. |

Coordinates are planar angular offsets, not RA/Dec values to be interpreted with a spherical-distance calculation. The image at HDU index `i` corresponds to source-position row `i`. Invalid shape measurements remain as NaN rows; each estimator applies its own finite-value mask during stacking. Filtering or sorting one input independently breaks the correspondence.

## Shape measurements

`shape_measurements.npz`, produced by the coursework command, contains `unweighted`, `weighted` and `hsm`. Each has shape `(n_sources, 2)` with the two component columns in catalogue order. Components are dimensionless. NaN rows indicate failed measurements. See the [method](method.md) for the distinct component conventions.

## Inference arrays

`analysis_results.npz` uses the following keys. Here `n_bins=10` and `n_grid=120` by default.

| Key | Shape / meaning |
| --- | --- |
| `radii_arcsec` | `(n_bins,)`; geometric radial-bin centres in arcsec |
| `mean_<method>` | `(n_bins,)`; mean tangential component |
| `counts_<method>` | `(n_bins,)`; number of finite halo–source pairs |
| `sum_squared_<method>` | `(n_bins,)`; sum of squared tangential components |
| `error_<method>` | `(n_bins,)`; standard error of the mean; NaN for fewer than two pairs |
| `halo_masses_msun_over_h` | `(n_grid,)`; $M_{200c}$ in $M_\odot/h$ |
| `source_redshifts` | `(n_grid,)`; dimensionless common source-redshift grid |
| `joint_log_likelihood` | `(n_grid, n_grid)`; rows are redshift, columns are mass |
| `joint_posterior` | Same shape; cell probabilities summing to one |
| `fixed_z_mass_posterior` | `(n_grid,)`; posterior density at $z_s=0.6$, integrating to one over mass |

Replace `<method>` with `unweighted`, `weighted` or `hsm`. Pair counts can exceed the source count because a source can contribute around multiple halos.

`profiles.csv` gives the same profile information in a flat table: `method`, `radius_arcsec`, `mean_tangential_shape`, `standard_error`, `pair_count`. The summary JSON records settings and grid maxima; it does not contain parameter credible intervals. `source_subset` identifies coursework runs made with `--max-sources`.

## Synthetic example

The synthetic catalogue is generated in memory. Its `truth.json` records the seed, Gaussian image profile, pixel noise, injected halo parameters and source counts. Synthetic notebook figures are labelled as such. The example is separate from the supplied coursework dataset.
