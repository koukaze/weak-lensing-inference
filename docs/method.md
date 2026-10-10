# Method and interpretation

## Image-shape estimators

All three estimators act on the same galaxy stamps. The direct moment methods first estimate the flux centroid. They form normalised quadrupole moments $Q_{xx}$, $Q_{yy}$ and $Q_{xy}$ about that centroid, then use

$$
D=Q_{xx}+Q_{yy}+2\sqrt{Q_{xx}Q_{yy}-Q_{xy}^2},\qquad
 e_1=\frac{Q_{xx}-Q_{yy}}{D},\qquad e_2=\frac{Q_{xy}}{D}.
$$

The second component is the convention used by the coursework implementation; the conventional reduced-shear moment definition has $2Q_{xy}$ in that numerator. Consequently the direct moment outputs are not calibrated equivalents of HSM components. Their profile comparison describes the implemented estimators, not an unbiased shear-calibration study.

The weighted estimator uses a circular Gaussian with FWHM 4 pixels, centred at the same unweighted flux centroid. Weighted moments are normalised by the weighted flux. Negative moment determinants and other non-finite outcomes produce invalid shape measurements.

HSM uses `galsim.hsm.FindAdaptiveMom` and the returned `observed_shape.g1` and `.g2`. These are observed image-shape parameters. The analysis uses them as shear proxies in the weak-lensing approximation and does not perform PSF correction, shear-response calibration or intrinsic-alignment modelling.

## Tangential stacking and uncertainty

For a source at polar angle $\phi$ about a halo, the tangential component is

$$
 g_t=-g_1\cos(2\phi)-g_2\sin(2\phi).
$$

Source and halo positions use a flat coordinate system in arcseconds. Each method independently removes rows with a non-finite shape component. Every remaining halo–source pair within $100\leq r<3600$ arcsec contributes to one of ten logarithmic radial bins. Bin centres are geometric means of the edges. A single source can contribute to multiple halos.

For a bin with $N>1$ pairs, the implemented standard error is

$$
 s^2=\max\left(0,\frac{\sum_i g_{t,i}^2-N\bar g_t^2}{N-1}\right),
 \qquad \sigma_{\bar g_t}=\sqrt{s^2/N}.
$$

Bins with zero or one pair have undefined errors and are excluded from inference. The estimate treats pair contributions as independent. It does not account for repeated-source correlations, correlations between radial bins or a survey covariance matrix.

## NFW prediction

The model uses GalSim's `NFWHalo` with concentration 4, halo redshift 0.3, $\Omega_m=0.3$ and $\Omega_\Lambda=0.7$. GalSim defines the mass as $M_{200c}$, in units of $M_\odot/h$. Here $h=H_0/(100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$; the numerical mass values must not be interpreted as plain solar masses without a value of $h$.

For each positive angular separation, the calculation evaluates the halo on its positive x-axis and returns minus the first component of `getShear(..., reduced=False)`. This yields the tangential shear $\gamma_t$. The prediction is evaluated at each bin centre, rather than averaged over the individual pair separations within that bin.

A single common mass and source redshift describe the stacked profile. The model comparison assumes that the HSM shape average approximates unreduced shear. Magnification, convergence corrections and a source-redshift distribution are outside this coursework calculation.

## Likelihood and posterior

The HSM profile enters a diagonal Gaussian likelihood:

$$
\log\mathcal L(M,z_s)=-\frac12\sum_b
\left[\frac{\bar g_{t,b}-\gamma_t(r_b;M,z_s)}{\sigma_b}\right]^2.
$$

Only bins with finite data, finite model predictions and finite positive errors enter the sum. The parameter-independent Gaussian normalisation is omitted. The command-line workflow rejects profiles without any informative bins.

The joint calculation uses 120 linearly spaced masses from $3.0\times10^{14}$ to $14.8\times10^{14}\,M_\odot/h$ and 120 redshifts from 0.31 to 0.8. Priors are uniform within this finite rectangle. Subtracting the maximum log likelihood stabilises exponentiation; normalising by the sum gives a probability per grid cell.

The 68% and 95% credible thresholds are obtained by sorting cell probabilities in descending order and accumulating their probability mass. They describe regions of this bounded, discretised posterior. If the grid is too coarse, thresholds can coincide and there need not be two distinct drawable contours. The reported best fit is the highest-probability grid cell.

For the one-dimensional calculation, $z_s=0.6$ is fixed and the same mass grid is used. Trapezoidal integration normalises the mass posterior as a density in $(M_\odot/h)^{-1}$. Fixing redshift removes one source of uncertainty; it does not remove the other image-measurement and model assumptions.

## Synthetic walkthrough

The example places sources at the radial-bin centres around one halo and draws their position angles uniformly. A circular Gaussian image is sheared using `g1=-gamma_t*cos(2*phi)` and `g2=-gamma_t*sin(2*phi)`, then independent Gaussian pixel noise is added. Assigning $\gamma$ directly as the GalSim shape parameter uses a weak-shear approximation, rather than a full lensing transformation.

There is no intrinsic ellipticity dispersion, PSF convolution, blending or source-redshift distribution. Placing sources at bin centres also removes bin-averaging mismatch by construction. This controlled example checks the connection between image shapes, stacking and inference; its parameter recovery is not a validation of survey-level mass accuracy.
