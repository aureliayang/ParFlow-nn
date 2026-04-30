# A Hybrid Physics–ML Framework for Integrating Groundwater Dynamics into Land Surface Modeling

## Status

The neural network surrogate model and CoLM202x are currently integrated and communicate smoothly.

---

## Repository Structure

### `clm_cbind`

Contains the `Makefile` and Fortran (`.F90`) source files used to compile the shared library `libclm_lsm.so`, which is invoked by the ParFlow surrogate model.

---

### `full_year_scripts`

Contains the Slurm job scripts and outputs for the full-year hybrid simulations presented in the paper.

---

### `other_inputs`

#### `ERA5_forcings`

48-hour ERA5-Land meteorological forcing data used for the test case.

#### `restart_files`

The test case is initialized from a hot start and requires restart files at the end of water year 2020 (WY2020).

#### `trained_model`

Pretrained surrogate model used in the hybrid simulation.

#### `static_inputs_combined46.pfb`

Static input file containing 46 land surface attributes.

#### `stats_press_evap2.yaml`

Normalization statistics (mean and standard deviation) for exchange fluxes and pressure head.


