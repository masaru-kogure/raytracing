## Author: Masaru Kogure
## Version: 3.1.0
## Email: masarukogure@yonsei.ac.kr
## Date: Last updated: 2026/04/09
## Latest bug-fix information
## The latest version fixes the main problems found in the first version.
## In the original code, there was a degree–radian inconsistency in the longitude update, the unpacking of
## dudx, dvdx, dudy, and dvdy was inconsistent, the RK4 stage construction was not self-consistent,
## and the Cs2 pathway could fail because the return structure changed depending on whether Cs2 was provided.
## In the latest version, the longitude update now uses a consistent radian-based form, the interpolation outputs are unpacked correctly,
## the RK4 treatment of trial wave numbers has been improved by separating the base values from the evaluation values, and Cs2 is now returned consistently.
## Overall, the latest code is much more consistent and reliable than the first version for the RK4 ray-tracing step.
## The latest program files are as follows:
## main_raytracing_rv2.py, derivation_v2.py, meteo_para_v3.py, runge_kutta4_v2.py, kernel_box.py, plot_ql.py
##
## I strongly recommend that you do not use the old version.
##
## Additional checks for the 20240124 JAWARA run:
## - main_raytracing_rv2.py now builds an increasing interpolation-time coordinate that matches the NetCDF time slice.
##   For the 20240124 11:47 UT launch, the background window is 2024-01-23 17:00 UT to 2024-01-24 12:00 UT,
##   and the model-time coordinate is -7 to 12 hours.
## - JAWARA W is stored as pressure velocity in hPa/s. The main program now converts it to Pa/s before calling
##   meteo_para.ver_wind(), which converts pressure velocity to geometric vertical velocity in m/s.
## - meteo_para.cori_para() now returns 2 * Omega * sin(latitude). The ray-tracing core already used the correct
##   Coriolis definition through runge_kutta4_v2.py.
## - derivation_v2.py includes fft_diff_axis(), an axis-wise FFT derivative. It was checked against fft_diff()
##   and is used to precompute horizontal wind gradients faster.
## - Each completed ray is saved immediately as a compressed npz file and a quick-look PNG under
##   /home/masarukogure/masarukogure/raytrance/20240124_rv2/.
## - The 20240124 run has been smoke-tested with real JAWARA fields through JAWARA loading, ray integration,
##   instability diagnostics, npz saving, and quick-look PNG saving.
##
## Known remaining caution:
## - The JAWARA fields used here contain -999 fill values, especially near the lowest model levels.
##   They can produce np.gradient warnings during Brunt-Vaisala-frequency calculation and log/density warnings
##   if a ray approaches the lowest valid background levels. For production analysis, mask fill values and/or stop
##   the ray at the lowest valid geometric-height level for the local column.
