# A Hybrid Physics–ML Framework for Integrating Groundwater Dynamics into Land Surface Modeling

## Status

The neural network surrogate model and CoLM202x are currently integrated and communicate smoothly.

## Repository Structure

**clm_cbind**: Makefile and Fortran (.F90) source files used to compile `libclm_lsm.so`, which is invoked by the ParFlow surrogate model.  

**full_year_scripts**: Slurm job scripts and outputs for the full-year hybrid simulations used in the paper.  

**other_inputs**:  

- **ERA5_forcings**: 48-hour ERA5-Land meteorological forcing data for the test case.  

- **restart_files**: Required for the hot-start initialization at the end of water year 2020 (WY2020).  

- **trained_model**: Pretrained surrogate model used in the hybrid system.  

- **static_inputs_combined46.pfb**: Static input file containing 46 land surface attributes.  

- **stats_press_evap2.yaml**: Mean and standard deviation for exchange fluxes and pressure head.