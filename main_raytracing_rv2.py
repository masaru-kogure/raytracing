##################################################
## Description
## This program calculates gravity-wave paths using a fourth-order Runge-Kutta method.
## It is translated from the IDL code presented in Kogure et al. (2018). 
## This code is maintained for Python 3.
##################################################
## Terms and Conditions of Use  
## This program is free to use for academic, non-commercial purposes. 
## Modification of the code is not recommended; any modifications are made at your own risk. 
## If used in publications, you must cite the specific references (see the following reference).
## We strongly encourage users to contact us for discussion before using the result of this software in publications, 
## to prevent misuse or misinterpretation of the output.
## The developers and their affiliated organizations are not responsible for any damages arising from use of the software.
## Reference
#### Raytracing theory
#### Marks, C. J., and S. D. Eckermann, 1995: A Three-Dimensional Nonhydrostatic Ray-Tracing Model for Gravity Waves: Formulation and Preliminary Results for the Middle Atmosphere. J. Atmos. Sci., 52, 1959–1984, https://doi.org/10.1175/1520-0469(1995)052<1959:ATDNRT>2.0.CO;2.
#### Growth rate of amplitudes
#### Kogure, M., Nakamura, T., Ejiri, M. K., Nishiyama, T., Tomikawa, Y., & Tsutsumi, M. (2018). Effects of horizontal wind structure on a gravity wave event in the middle atmosphere over Syowa (69°S, 40°E), the Antarctic. Geophysical Research Letters, 45, 5151–5157. https://doi.org/10.1029/2018GL078264
##################################################
## Author: Masaru Kogure
## Version: 2.1.0
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2026/04/09
## The latest bugfix information can be found in "read_me.txt".
##################################################
from random import weibullvariate
import numpy as np
from derivation_v2 import derivation
from meteo_para_v3 import meteo_para
from kernel_box import kernel_box
from scipy import signal
from netCDF4 import Dataset
import netCDF4 as nc
import glob
from pathlib import Path
from runge_kutta4_v2  import runge_kutta4
from netCDF4 import num2date
import cftime
from plot_ql  import plot_ql 

# Case setup. The ray starts at 11:47 UT and is integrated backward.
date = "20240124"
dt = -60.
her = 11.
mn = 47.
timeM0 = her + mn/60.
num_time_records = 20
data_end_hour = int(np.ceil(timeM0))
target_time = cftime.DatetimeGregorian(int(date[0:4]), int(date[4:6]), int(date[6:9]), data_end_hour, 0, 0, 0, has_year_zero=False)
gome = []
kh_list = ["259", "333", "386", "364"] 
k_list = np.array([-0.003862308695022451, -0.0030040178739063348 , -0.0017564402810304638,-0.0025884383088869765]) * 2 * np.pi * 1.e-3
l_list = np.array([0., 0.0003567860710717871, 1.430697973529854e-16, -0.0009174311926605055]) * 2 * np.pi * 1.e-3
lonMlist = [132, 132, 138, 136]
latMlist = [45, 49, 45, 48]
pltnum = np.array(len(lonMlist))
lambdaz = [15, 20, 25, 30, 35, 40]
Tamp = np.array(lambdaz) * 0 + 0.1
zini = 87 * 1e3

#----------------constants----------------------
global const, O2, r, dt7200, dt3600, dt_2, g0
const = 29.26 # m/K
O2 = 0.00014584231#7.2921159 * 1e-5 * 2 #rad/s
r = 6.3781 * 1.e6 #[m] Mean radius of the earth
dt3600 =  dt/3600.
dt_2 = dt  * 0.5
g0 = 9.80665 #[m/s^2]
#----------------constants----------------------
variname = ['T', 'U', 'V', 'W', 'Z']
#--------read JAWARA data-----------------
i_date = 0
year = date[0:4]
mm = date[4:6]
dd = date[6:8]
# Keep the interpolation coordinate in the same chronological order as the NetCDF slice.
time1 = data_end_hour - np.arange(num_time_records - 1, -1, -1, dtype=float)
filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/JAWARA/" + year + "/T" + year[2:5] + mm + "*" ))
ncfileJ = Dataset(filenameJ[0])

lat = np.squeeze(np.array(ncfileJ.variables['latitude'][:]))
lon = np.squeeze(np.array(ncfileJ.variables['longitude'][:]))
press = np.squeeze(np.array(ncfileJ.variables['level']))[0:-1] * 1.e2
time_var = ncfileJ.variables["time"]
t_num = time_var[:] 
units = getattr(time_var, "units", None)
calendar = getattr(time_var, "calendar", "standard")
t_dt = num2date(t_num, units=units, calendar=calendar)
target_index = np.where(np.array(t_dt) == target_time)[0]
if target_index.size == 0:
    raise ValueError(f"{target_time} was not found in {filenameJ[0]}")
ihh = int(target_index[0] + 1)
t_dt = t_dt[ihh-len(time1):ihh]

t = np.squeeze(np.array(ncfileJ.variables["t"][ihh-len(time1):ihh, 0:-1, :, :]))
ncfileJ.close()
filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/JAWARA/" + year + "/U" + year[2:5] + mm + "*" ))
ncfileJ = Dataset(filenameJ[0])
u = np.squeeze(np.array(ncfileJ.variables["u"][ihh-len(time1):ihh, 0:-1, :, :]))
ncfileJ.close()
filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/JAWARA/" + year + "/V" + year[2:5] + mm + "*" ))
ncfileJ = Dataset(filenameJ[0])
v = np.squeeze(np.array(ncfileJ.variables["v"][ihh-len(time1):ihh, 0:-1, :, :]))
filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/JAWARA/" + year + "/W" + year[2:5] + mm + "*" ))
ncfileJ = Dataset(filenameJ[0])
w = np.squeeze(np.array(ncfileJ.variables["w"][ihh-len(time1):ihh, 0:-1, :, :]))
filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/JAWARA/" + year + "/Z" + year[2:5] + mm + "*" ))
ncfileJ = Dataset(filenameJ[0])
z = np.squeeze(np.array(ncfileJ.variables["z"][ihh-len(time1):ihh, 0:-1, :, :]))
ncfileJ.close()
ncfileJ = Dataset(filenameJ[0])
ncfileJ.close()
#--------read JAWARA data-----------------
press = np.transpose(np.tile(press, (1,1,1,1)), (0,3,1,2)) 
PT = meteo_para.poten_temp(t, press * 1.e-2)
NF = meteo_para.brunt_fre(z, PT, dim = 4)
rho = meteo_para.density(press, t)
# JAWARA W is stored as hPa/s. Convert it to Pa/s before converting to m/s.
wv = meteo_para.ver_wind(w * 1.e2, rho)
H = meteo_para.scale_p_h(t)
phi = lat/180 * np.pi
ram = lon/180 * np.pi
dlat = lat[-2] - lat[-1]
dlon = lon[1] - lon[0]
dram = np.abs((lon[1] - lon[0])/180 * np.pi * r)
dphi = np.abs((lat[1] - lat[0])/180 * np.pi * r)
#----------------derived fields-----------------
dudx = u * 0
dvdx = v * 0
NERA = t.shape
wc = kernel_box.cosine_roll(len(lat), 0.1)
figuresave = '/home/masarukogure/masarukogure/raytrance/'
output_dir = Path(figuresave) / f"{date}_rv2"
data_save_dir = output_dir / "npz"
ql_save_dir = output_dir / "ql"
data_save_dir.mkdir(parents=True, exist_ok=True)
ql_save_dir.mkdir(parents=True, exist_ok=True)

# Precompute horizontal wind gradients once; RK4 interpolation samples these fields repeatedly.
for i_lat in range(NERA[2]):
    dram1 = dram * np.abs(np.cos(phi[i_lat]))
    dudx[:, :, i_lat, :] = derivation.fft_diff_axis(u[:, :, i_lat, :], dram1, axis=2, idel_filter=1/(2 * dphi))
    dvdx[:, :, i_lat, :] = derivation.fft_diff_axis(v[:, :, i_lat, :], dram1, axis=2, idel_filter=1/(2 * dphi))

dudy = derivation.fft_diff_axis(u * wc.reshape(1, 1, -1, 1), dphi, axis=2)
dvdy = derivation.fft_diff_axis(v * wc.reshape(1, 1, -1, 1), dphi, axis=2)



op_min = 0
cc_min = 0

for ilambdaz, iTamp in zip(lambdaz, Tamp):
    for kh, k, l in zip(kh_list, k_list, l_list):
        for lonM, latM  in zip(lonMlist, latMlist):
            # Copy the launch point because lonM/latM are expanded into trajectory arrays below.
            lonMs = lonM
            latMs = latM
            if (lonMs == lonMlist[0]) and (latMs == latMlist[0]) :
                legend1 = [str(lonMs)+' E, ' + str(latMs) + ' N' ]
            timeM = timeM0
            zM = zini
            m1 = -2*np.pi/(ilambdaz * 1e3)
            Tin = iTamp
            lonM = lonMs
            latM = latMs
            lonM1 = lonM
            latM1 = latMs
            zM1 = zM
            kM = k
            lM = l
            k1 = kM
            l1 = lM
            timeM1 = timeM

            # Interpolate the background state at the launch point and infer the absolute frequency.
            f1, b1, dudx1, dvdx1, dudy1, dvdy1, u1, v1, w1, NF1, H1, Cs21 = runge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time1, z, lonM, latM, timeM, zM, dlat, dlon)
            ome = np.sqrt(meteo_para.cal_dis_ral( k1, l1, [], f1, NF1, H1, m = m1))
            gome = ome + (u1 * k1 + v1 * l1)
            print(lonM, latM)

            i = 0
            uM = np.NAN
            vM = np.NAN
            wM = np.NAN
            NFM = np.NAN
            HM = np.NAN
            omeM = np.NAN
            dz = 0.1 
            # Stop when the ray reaches the ground or the model returns a NaN step.
            while (not np.isnan(dz)) and zM1 >0:
                dy, dx, dz, dk, dl, u1, v1, w1, NF1, ome1, H1,cs21 = runge_kutta4.main_runge(dudx, dvdx, dudy, dvdy, u, v, wv, NF, H, gome, k1, l1, lon, lat, time1, z, lonM1, latM1, timeM1, zM1, phi, ram, dt, dlat, dlon)
                lonM1 = lonM1 + dx
                latM1 = latM1 + dy
                zM1 = zM1 + dz 
                k1 = k1 + dk
                l1 = l1 + dl
                timeM1 = timeM1 + dt3600 
                lonM = np.append(lonM, lonM1)
                latM = np.append(latM, latM1)
                zM = np.append(zM, zM1 )
                kM = np.append(kM, k1)
                lM = np.append(lM, l1)
                timeM = np.append(timeM, timeM1)
                if i == 0 :
                    uM = u1
                    vM = v1
                    wM = w1
                    NFM = NF1
                    HM = H1
                    omeM = ome1
                else:
                    uM = np.append(uM, u1)
                    vM = np.append(vM, v1)
                    wM = np.append(wM, w1)
                    NFM = np.append(NFM, NF1)
                    HM = np.append(HM, H1)
                    omeM = np.append(omeM, ome1)        
                i = i + 1
            lonM = lonM[0:-1]
            latM = latM[0:-1]
            zM = zM[0:-1]
            kM = kM[0:-1]
            lM = lM[0:-1]
            timeM = timeM[0:-1]
            # Evaluate simple instability diagnostics along the completed ray path.
            N_full, Ri, mM = runge_kutta4.instability(Tin, NFM, uM, vM, zM, HM, latM, lonM, timeM, omeM, kM, lM, m1, rho, t, z, dlat, dlon, time1, lat, lon )
            sigma = np.abs(np.gradient(mM, zM)/(mM**2))
            mmo1 = np.abs(2 * np.pi/mM * 1.e-3)
            nd1 = (mmo1 > 1)
            dir = 90 - (np.arctan2(lM, kM)/(np.pi) * 180)
            # Save every completed ray immediately so a long sweep still leaves partial results.
            case_id = plot_ql.ray_case_id(date, kh, ilambdaz, lonMs, latMs)
            npz_path = plot_ql.save_ray_npz(
                data_save_dir, case_id,
                date=date, kh=kh, lambdaz=ilambdaz, Tamp=Tin,
                lon0=lonMs, lat0=latMs, zini=zini, dt=dt,
                timeM0=timeM0, gome=gome,
                lon=lonM, lat=latM, z=zM, time=timeM,
                k=kM, l=lM, m=mM, vertical_wavelength_km=mmo1,
                direction=dir, u=uM, v=vM, w=wM, N2=NFM,
                H=HM, omega_intrinsic=omeM, N2_full=N_full,
                Ri=Ri, sigma=sigma,
            )
            ql_path = plot_ql.plot_ql_ray(ql_save_dir, case_id, lonM, latM, zM, timeM, uM, vM, NFM, Ri, mmo1, dir)
            print(f"saved {npz_path}")
            print(f"saved {ql_path}")
            print('end')

print('end')
                        
