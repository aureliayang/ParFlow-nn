
MODULE clm_lsm_c_interface
  USE ISO_C_BINDING
IMPLICIT NONE
END MODULE clm_lsm_c_interface

SUBROUTINE clm_lsm_c(pressure, saturation, evap_trans, topo, porosity, pf_dz_mult, istep_pf, dt, time, &
  start_time_pf, pdx, pdy, pdz, ix, iy, nx, ny, nz, nx_f, ny_f, nz_f, nz_rz, ip, npp, npq, npr, gnx, gny, rank, &
  sw_pf, lw_pf, prcp_pf, tas_pf, u_pf, v_pf, patm_pf, qatm_pf, lai_pf, sai_pf, z0m_pf, displa_pf, slope_x_pf, slope_y_pf, &
  eflx_lh_pf, eflx_lwrad_pf, eflx_sh_pf, eflx_grnd_pf, qflx_tot_pf, qflx_grnd_pf, qflx_soi_pf, qflx_eveg_pf, qflx_tveg_pf, &
  qflx_in_pf, swe_pf, t_g_pf, t_soi_pf, clm_dump_interval, clm_1d_out, clm_forc_veg, clm_output_dir, clm_output_dir_length, &
  clm_bin_output_dir, write_CLM_binary, slope_accounting_CLM, beta_typepf, veg_water_stress_typepf, wilting_pointpf, &
  field_capacitypf, res_satpf, irr_typepf, irr_cyclepf, irr_ratepf, irr_startpf, irr_stoppf, irr_thresholdpf, qirr_pf, &
  qirr_inst_pf, irr_flag_pf, irr_thresholdtypepf, soi_z, clm_next, clm_write_logs, clm_last_rst, clm_daily_rst, &
  pf_nlevsoi, pf_nlevlak) BIND(C)

  USE ISO_C_BINDING
  IMPLICIT NONE

  REAL(C_DOUBLE), INTENT(IN) :: pressure(*)
  REAL(C_DOUBLE), INTENT(IN) :: saturation(*)
  REAL(C_DOUBLE), INTENT(OUT) :: evap_trans(*)
  REAL(C_DOUBLE), INTENT(IN) :: topo(*)
  REAL(C_DOUBLE), INTENT(IN) :: porosity(*)
  REAL(C_DOUBLE), INTENT(IN) :: pf_dz_mult(*)
  INTEGER(C_INT), VALUE :: istep_pf
  REAL(C_DOUBLE), VALUE :: dt
  REAL(C_DOUBLE), VALUE :: time
  REAL(C_DOUBLE), VALUE :: start_time_pf
  REAL(C_DOUBLE), VALUE :: pdx, pdy, pdz
  INTEGER(C_INT), VALUE :: ix, iy, nx, ny, nz, nx_f, ny_f, nz_f, nz_rz
  INTEGER(C_INT), VALUE :: ip, npp, npq, npr, gnx, gny, rank
  REAL(C_DOUBLE), INTENT(IN) :: sw_pf(*), lw_pf(*), prcp_pf(*), tas_pf(*), u_pf(*), v_pf(*), patm_pf(*), qatm_pf(*)
  REAL(C_DOUBLE), INTENT(IN) :: lai_pf(*), sai_pf(*), z0m_pf(*), displa_pf(*), slope_x_pf(*), slope_y_pf(*)
  REAL(C_DOUBLE), INTENT(OUT) :: eflx_lh_pf(*), eflx_lwrad_pf(*), eflx_sh_pf(*), eflx_grnd_pf(*)
  REAL(C_DOUBLE), INTENT(OUT) :: qflx_tot_pf(*), qflx_grnd_pf(*), qflx_soi_pf(*), qflx_eveg_pf(*), qflx_tveg_pf(*), qflx_in_pf(*)
  REAL(C_DOUBLE), INTENT(OUT) :: swe_pf(*), t_g_pf(*), t_soi_pf(*)
  INTEGER(C_INT), VALUE :: clm_dump_interval, clm_1d_out, clm_forc_veg
  CHARACTER(KIND=C_CHAR), DIMENSION(*) :: clm_output_dir
  INTEGER(C_INT), VALUE :: clm_output_dir_length, clm_bin_output_dir, write_CLM_binary, slope_accounting_CLM, beta_typepf
  INTEGER(C_INT), VALUE :: veg_water_stress_typepf
  REAL(C_DOUBLE), VALUE :: wilting_pointpf, field_capacitypf, res_satpf
  INTEGER(C_INT), VALUE :: irr_typepf, irr_cyclepf
  REAL(C_DOUBLE), VALUE :: irr_ratepf, irr_startpf, irr_stoppf, irr_thresholdpf
  REAL(C_DOUBLE), INTENT(OUT) :: qirr_pf(*), qirr_inst_pf(*)
  REAL(C_DOUBLE), INTENT(IN) :: irr_flag_pf(*)
  INTEGER(C_INT), VALUE :: irr_thresholdtypepf
  INTEGER(C_INT), VALUE :: soi_z
  INTEGER(C_INT), VALUE :: clm_next, clm_write_logs, clm_last_rst, clm_daily_rst, pf_nlevsoi, pf_nlevlak

  CALL CLM_LSM(pressure, saturation, evap_trans, topo, porosity, pf_dz_mult, istep_pf, dt, time, &
    start_time_pf, pdx, pdy, pdz, ix, iy, nx, ny, nz, nx_f, ny_f, nz_f, nz_rz, ip, npp, npq, npr, gnx, gny, rank, &
    sw_pf, lw_pf, prcp_pf, tas_pf, u_pf, v_pf, patm_pf, qatm_pf, lai_pf, sai_pf, z0m_pf, displa_pf, slope_x_pf, slope_y_pf, &
    eflx_lh_pf, eflx_lwrad_pf, eflx_sh_pf, eflx_grnd_pf, qflx_tot_pf, qflx_grnd_pf, qflx_soi_pf, qflx_eveg_pf, qflx_tveg_pf, &
    qflx_in_pf, swe_pf, t_g_pf, t_soi_pf, clm_dump_interval, clm_1d_out, clm_forc_veg, clm_output_dir, clm_output_dir_length, &
    clm_bin_output_dir, write_CLM_binary, slope_accounting_CLM, beta_typepf, veg_water_stress_typepf, wilting_pointpf, &
    field_capacitypf, res_satpf, irr_typepf, irr_cyclepf, irr_ratepf, irr_startpf, irr_stoppf, irr_thresholdpf, qirr_pf, &
    qirr_inst_pf, irr_flag_pf, irr_thresholdtypepf, soi_z, clm_next, clm_write_logs, clm_last_rst, clm_daily_rst, pf_nlevsoi, &
    pf_nlevlak)

END SUBROUTINE clm_lsm_c
