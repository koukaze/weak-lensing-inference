# References

## Model and data sources

S. Gratton, N. MacCrann, F. McCarthy and B. Sherwin, *Cosmology*: specification for the simulated-data analysis, radial bins, halo parameters and inference tasks. The supplied data must remain outside the repository under the data-use instructions.

## NFW halo model

J. F. Navarro, C. S. Frenk and S. D. M. White, *A Universal Density Profile from Hierarchical Clustering*, The Astrophysical Journal **490**, 493-508 (1997).
[Article](https://doi.org/10.1086/304888) · [Preprint](https://arxiv.org/abs/astro-ph/9611107)

The analysis uses an NFW density profile for the foreground halos, evaluated through GalSim's lensing implementation.

## GalSim

B. Rowe, M. Jarvis, R. Mandelbaum et al., *GalSim: The modular galaxy image simulation toolkit*.
[Preprint](https://arxiv.org/abs/1407.7676) · [Software](https://github.com/GalSim-developers/GalSim)

- [HSM documentation](https://galsim-developers.github.io/GalSim/_build/html/hsm.html): `FindAdaptiveMom` measures adaptive image moments and returns `observed_shape`. This analysis uses its `g1` and `g2` components.
- [NFW halo documentation](https://galsim-developers.github.io/GalSim/_build/html/nfwhalo.html): `NFWHalo` defines halo mass within an overdensity of 200 times the critical density, in units of $M_\odot/h$. Angular positions are in arcseconds. Calling `getShear` with `reduced=False` returns shear rather than reduced shear.

The course authors supplied the simulated galaxy stamps, source and halo positions, and the `nfw_theory.py` helper. This package evaluates NFW shear through GalSim's public `NFWHalo.getShear` interface.
