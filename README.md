# Qualitative imaging of orthotropic elastic plates by first-iteration adjoint sensitivity maps: finite-element code for the reflection–transmission residual-wave interaction indicator

This repository contains the computer code that produced the numerical results of the article

> H. Hafidi Alaoui, H. Zayeni, *Qualitative imaging of orthotropic elastic plates by first-iteration adjoint sensitivity maps: a reflection–transmission residual-wave interaction indicator*, Wave Motion 148 (2027) 103863. https://doi.org/10.1016/j.wavemoti.2026.103863

Version 1 of the code is archived on Zenodo: https://doi.org/10.5281/zenodo.23534379

Version 1 preserves the computational implementation used for the article. Changes made for publication are limited to comments, documentation and descriptive text (command-line help, report metadata and console messages); no computation has been changed.

## Contents

| Path | Content |
|---|---|
| `elastic_fwi/` | Python package: FEniCSx/PETSc time-domain elastodynamic solver for orthotropic plates, direct and adjoint problems, material sensitivity kernels, residual-wave interaction indicator, campaign drivers, diagnostics and post-processing |
| `elastic_fwi/campaigns/run_gradient_combination_campaign.py` | Campaign driver and presets used for the article |
| `elastic_fwi/tests/` | Unit and integration tests (pytest) |
| `scripts/revision_wave_motion/` | Analysis, verification and figure scripts that read campaign outputs |
| `configs/defects/` | Defect configurations |

## Software environment

The campaigns were run with the following versions (conda environment):

| Package | Version |
|---|---|
| Python | 3.11.15 |
| DOLFINx | 0.10.0 |
| Basix | 0.10.0 |
| UFL | 2025.2.1 |
| petsc4py | 3.24.4 |
| mpi4py | 4.1.1 |
| Gmsh | 4.15.2 |
| meshio | 5.3.5 |
| NumPy | 2.4.2 |
| SciPy | 1.17.1 |
| pandas | 3.0.3 |
| VTK | 9.6.1 |
| h5py | 3.15.1 |
| pytest | 9.0.3 |

The published figures were rendered in a separate environment with Python 3.13.12, Matplotlib 3.11.0 and NumPy 2.5.0. Rendering the figures with another Matplotlib version gives visually equivalent figures that are not identical byte for byte.

The symbolic verification scripts (`verify_mirror_symmetry_symbolic.py`, `derive_normal_strain_curvature_asymmetry.py`) additionally require SymPy.

Install the package in development mode from the repository root:

```bash
pip install -e .
```

## Reproducing the central campaigns

The two central campaigns of the article use the receiver configurations `line` and `box`. They were run with 32 MPI processes:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

COMMON="--dt 0.025 --num-steps 800 --every-k 1 --source-num-periods 1.0 \
  --export-initial-state --export-final-state --visual-output-format pvd \
  --gradient-export-mode both --ksp-type cg --pc-type gamg \
  --ksp-rtol 1e-8 --ksp-atol 1e-10 --ksp-max-it 500 --clean"

mpirun -np 32 python -u -m elastic_fwi.campaigns.run_gradient_combination_campaign \
  --preset article_L73_gmsh_p1_ppw15_near40_T20_no_absorbing \
  --output-dir outputs/line $COMMON

mpirun -np 32 python -u -m elastic_fwi.campaigns.run_gradient_combination_campaign \
  --preset article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T20_no_absorbing \
  --output-dir outputs/box $COMMON
```

On the reference workstation (AMD Ryzen Threadripper PRO, 32 MPI processes), the `line` campaign took about 16.5 h and the `box` campaign about 29 h. Together the two campaigns write about 2 TB of output, mostly VTK time series.

A complete re-run of both campaigns with this version of the code reproduced the archived results: 385 quantities were compared, none failed, 209 were identical bit for bit, and the largest relative difference was 1.1e-9 (threshold 1e-6). The post-processing steps that read the campaign fields were re-run on these outputs and agree within the same threshold, with one exception. In the scattered-wavefield reconstruction (`reconstruct_scattered_wavefield.py`), 113 values evaluated before the wave reaches the defect differ by more than 1e-6 in relative terms (largest relative difference 3.7e-6). At those times the scattered field is solver noise: the values concerned are at most 2.6e-8 of the maximum of their column, and the difference relative to that maximum does not exceed 3.4e-13. All values from the arrival of the wave at the defect onwards agree within the threshold.

## Tests

```bash
pytest
```

## Citation

If you use this code, please cite the article above and the Zenodo archive. Citation metadata are given in `CITATION.cff`.

## Licence

See `LICENSE`.
