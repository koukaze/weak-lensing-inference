# Running the analysis

Run commands from the repository root. Python **3.11+** is required; the pinned environment was tested with Python **3.13.7** and GalSim **2.8.5**. No GPU is needed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

In Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The requirements file pins direct dependencies; it is not a full transitive lockfile.

## Synthetic example

```bash
python -m weak_lensing.demo --output-dir outputs/synthetic --seed 625
```

The default example places 200 sources at each of the ten radial-bin centres around one halo. It creates 50 × 50 pixel images, measures all three shape estimators and evaluates the same inference grid used for the coursework. The seed controls source angles and additive pixel noise. Reusing the seed and dependency versions gives reproducible inputs; changing the sample count changes the random sequence assigned to images.

The injected mass is $5\times10^{14}\,M_\odot/h$ and the source redshift is 0.6. The input shear is assigned directly as an image-shape parameter under the weak-shear approximation. This is a toy calculation with circular intrinsic galaxies, no PSF and no convergence. It does not reproduce the supplied coursework simulation.

## Coursework inputs

Point `A6_DATA_DIR` to the directory containing the supplied inputs:

```bash
export A6_DATA_DIR="/path/to/a6/data"
python scripts/run_analysis.py --output-dir outputs/coursework
```

Or pass the path directly:

```bash
python scripts/run_analysis.py --data-dir "/path/to/a6/data" --output-dir outputs/coursework
```

The directory must contain `galaxy_stamps.fits`, `source_positions.txt` and `halo_positions.txt`. The FITS file is about 1.44 GB. Each source-position row must correspond to the image HDU with the same zero-based index, including the primary HDU. See [Data format and outputs](docs/data.md). The program reads the inputs in place and processes images in batches of 256; it does not copy them into the repository.

Defaults reproduce the coursework settings: all sources, all halos, ten radial bins and 120 samples along each inference axis. Use a smaller input subset only for a software check:

```bash
python scripts/run_analysis.py --max-sources 2000 --grid-size 30 --output-dir outputs/check
```

This selects the first 2,000 sources with their corresponding stamps and keeps all halo positions. It does not reproduce the full-data posterior. `--batch-size` changes the measurement batch size without changing the estimator or catalogue order. Runtime and memory depend on the FITS reader and machine; allow several minutes for the full input file.

## Generated files

| File | Contents |
| --- | --- |
| `analysis_results.npz` | Bin centres, profiles, counts, errors, parameter grids and posterior arrays |
| `summary.json` | Input counts, failed-shape counts, settings and grid maxima |
| `profiles.csv` | One row per method and radial bin |
| `shear_profiles.png` / `.svg` | Three stacked profiles with standard errors |
| `joint_posterior.png` / `.svg` | Joint mass–redshift posterior and credible contours |
| `fixed_redshift_mass.png` / `.svg` | Mass posterior at source redshift 0.6 |
| `shape_measurements.npz` | Coursework command: one shape array per estimator, in input order |
| `truth.json` | Synthetic command: generating parameters and seed |

Generated files go under `outputs/`, which is excluded from version control. Commands overwrite matching filenames in their output directory; use separate directories to retain different runs.

## Notebooks

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name weak-lensing --display-name "Python (weak lensing)"
jupyter lab notebooks/01_synthetic_walkthrough.ipynb
```

Select **Python (weak lensing)** and run all cells from a fresh kernel. The first notebook generates its own inputs and includes the figure shown in the README.

The second notebook supports two modes. For a fresh coursework analysis, set `A6_DATA_DIR` before starting Jupyter. To inspect the numerical results of a completed command-line run without repeating image measurement:

```bash
export A6_RESULTS_DIR="$(pwd)/outputs/coursework"
jupyter lab notebooks/02_coursework_analysis.ipynb
```

`A6_RESULTS_DIR` takes precedence over `A6_DATA_DIR`. Unset it before requesting a fresh run. If neither is set, the notebook explains the required input and skips the data-dependent cells.

## Tests and interpretation

```bash
python -m pytest -q
python -m pip check
```

The tests cover analytic image moments, HSM shapes, tangential projection, bin boundaries, failed rows, uncertainties, posterior normalisation, the NFW prediction and batched FITS processing. A successful run verifies software behaviour; interpretation still depends on the assumptions in [Method and interpretation](docs/method.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named weak_lensing` | Activate the environment, install the package and select its notebook kernel. |
| Missing coursework files | Set `A6_DATA_DIR` to the external input directory, or use the synthetic example. |
| Image/position count mismatch | Preserve every FITS HDU and catalogue row in their supplied order. |
| No valid HSM inference bins | Check source coordinates, stamp quality, sample size and the 100–3,600 arcsec radial range. |
| Sparse credible contours with a small grid | Increase `--grid-size`; a posterior concentrated in one cell may have no distinct 68% and 95% contour. |
| Different inferred masses | Compare input subset, grid size and dependency versions; masses use $M_\odot/h$. |
