from random import weibullvariate
import numpy as np
from derivation import derivation
from meteo_para_v2 import meteo_para
from lunge_kutta4 import lunge_kutta4
from kernel_box import kernel_box
from scipy import signal
from netCDF4 import Dataset
import netCDF4 as nc
import matplotlib.pyplot as plt
import glob
from lunge_kutta4 import lunge_kutta4
import pymsis

date = "20210812"
dt = -60.
her = 23
mn = 0.
ihh = ((np.int32(date[6::])) - 1) + (np.int32(her) - 1)
gome = []


#----------------read data----------------
i_date = 0
filename = "/RAID0/home/kogure/data/ERA5/physics/ERA5_"+ date[::6] + ".nc"#test
nd = filename.find('2021')
year = date[0:4]
mm = date[4:6]
dd = date[6:8]
time1 = 3 *  np.arange(16)
kh_list = ["170", "210"] 
k_list = 2 * np.pi *  np.array([ np.cos(np.pi/2), np.cos(np.pi/4), np.cos(0), np.cos(-np.pi/4)])
l_list = 2 * np.pi * np.array([ np.sin(np.pi/2), np.sin(np.pi/4), np.sin(0), np.sin(-np.pi/4)])

zini = 300 * 1e3
lonMlist = np.array([131, 133.8, 135, 133.5])
latMlist = np.array([37, 36., 33, 31.])
pltnum = np.array(len(lonMlist))
fre = 2 * np.pi/np.array([17 * 60])
#----------------para---------------------------
global const, O2, r, dt7200, dt3600, dt_2, g0
const = 29.26 # m/K
O2 = 0.00014584231#7.2921159 * 1e-5 * 2 #rad/s
r = 6.3781 * 1.e6 #[m] Mean radius of the earth
#dt7200 =  dt/7200.
dt3600 =  dt/3600.
dt_2 = dt  * 0.5
g0 = 9.80665 #[m/s^2]
R0 = 8.3134 * 1.e3 #[J/(K*mol)] gas constant
#----------------para---------------------------
variname = ['T', 'U', 'V', 'W', 'Z']
#--------read WACCM-X data--------------------------
filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/WACCM-X/FXSD.f09_f09_mg17.001.cam.h1." + year + "-" + mm + "-" + dd + "-00000.nc_subset" ))
ncfileJ = Dataset(filenameJ[0])
lat = np.squeeze(np.array(ncfileJ.variables['lat'][:]))
lon = np.squeeze(np.array(ncfileJ.variables['lon'][:]))
#lonJ[lonJ > 180] = lonJ[lonJ > 180] - 360 
press = np.squeeze(np.array(ncfileJ.variables['lev'])) * 1.e2
t = np.squeeze(np.array(ncfileJ.variables["T"]))
u = np.squeeze(np.array(ncfileJ.variables["U"]))
v = np.squeeze(np.array(ncfileJ.variables["V"]))
w = np.squeeze(np.array(ncfileJ.variables["OMEGA"]))
z = np.squeeze(np.array(ncfileJ.variables["Z3"]))
ncfileJ.close()

filenameJ = sorted(glob.glob("/home/masarukogure/masarukogure/data/WACCM-X/FXSD.f09_f09_mg17.001.cam.h1." + year + "-" + mm + "-" + str(np.int16(dd) + 1).zfill(2) + "-00000.nc_subset" ))
ncfileJ = Dataset(filenameJ[0])
t = np.append(t, np.squeeze(np.array(ncfileJ.variables["T"])), axis = 0)
u = np.append(u, np.squeeze(np.array(ncfileJ.variables["U"])), axis = 0)
v = np.append(v, np.squeeze(np.array(ncfileJ.variables["V"])), axis = 0)
w = np.append(w, np.squeeze(np.array(ncfileJ.variables["OMEGA"])), axis = 0)
z = np.append(z, np.squeeze(np.array(ncfileJ.variables["Z3"])), axis = 0)
ncfileJ.close()
#--------read WACCM-X data--------------------------
phi = lat/180 * np.pi
ram = lon/180 * np.pi
dlat = np.abs(lat[-2] - lat[-1])
dlon = np.abs(lon[1] - lon[0])
dram = (lon[1] - lon[0])/180 * np.pi * r
dphi = (lat[1] - lat[0])/180 * np.pi * r
ilon = 53
ilat = 65
#----------------read data----------------
#---------------------msis----------------
# geomagnetic_activity=-1 is a storm-time run
print(np.int32(her))
g = g0 * (r ** 2 / (z + r)**2)
dates = np.datetime64(year+"-"+mm+"-"+dd+"T"+str(np.int32(her)).zfill(2)+":00")
alt = np.mean( z[:, :,ilon, ilat], axis = 0) * 1.e-3
data = np.squeeze(pymsis.calculate(dates, lon[ilon], lat[ilat], alt, geomagnetic_activity=-1))
R = np.nansum([data[:,1]/28.01, data[:,3]/16, data[:,2]/32, data[:,7]/14, data[:,9]/30], axis = 0 )/np.nansum([data[:,1], data[:,3], data[:,2], data[:,7], data[:,9]], axis = 0 ) * R0#+ CO2/44.01)
Cv = np.nansum([5/2 * data[:,1]/28.01, 3/2 *  data[:,3]/16, 5/2 * data[:,2]/32,  3/2 * data[:,7]/14, 5/2 * data[:,9]/30], axis = 0 )/np.nansum([data[:,1], data[:,3], data[:,2], data[:,7], data[:,9]], axis = 0 ) * R0#+ CO2/44.01)
Cp = np.nansum([ 7/2 * data[:,1]/28.01, 5/2 * data[:,3]/16, 7/2 * data[:,2]/32, 5/2 *  data[:,7]/14,  7/2 * data[:,9]/30], axis = 0 )/np.nansum([data[:,1], data[:,3], data[:,2], data[:,7], data[:,9]], axis = 0 ) * R0#+ CO2/44.01)
R = np.transpose(np.tile(R, (1,1,1,1)), (0,3,1,2)) 
Cv = np.transpose(np.tile(Cv, (1,1,1,1)), (0,3,1,2))  
Cp = np.transpose(np.tile(Cp, (1,1,1,1)), (0,3,1,2))   
press = np.transpose(np.tile(press, (1,1,1,1)), (0,3,1,2)) 
PT = meteo_para.poten_temp(t, press * 1.e-2, R = R)
NF = meteo_para.brunt_fre(z, PT, dim = 4)
rho = meteo_para.density(press, t, R= R)
wv = meteo_para.ver_wind(w, rho)
H = meteo_para.scale_h(t, R = R, g = g)
Cs2 = meteo_para.sound_wave_speed(t, Cp = Cp, R = R, Cv = Cv)
#---------------------msis----------------

dudx = u * 0
dvdx = v * 0
dudy = u * 0
dvdy = v * 0
NERA = t.shape
wc = kernel_box.cosine_roll(len(lat), 0.1)

for i_high in range(NERA[1]):
    for i_time in range(NERA[0]):
        for i_lat in range(NERA[2]):
            dram1 = dram * np.abs(np.cos(phi[i_lat]))
            dudx[i_time, i_high, i_lat,:] = derivation.fft_diff(u[i_time, i_high, i_lat,:], dram1, idel_filter=1/(2 * dphi))
            dvdx[i_time, i_high, i_lat,:] = derivation.fft_diff(v[i_time, i_high, i_lat,:], dram1, idel_filter=1/(2 * dphi))
        for i_lon in range(NERA[3]):
            dudy[i_time, i_high, :, i_lon] = derivation.fft_diff(u[i_time, i_high, :, i_lon] * wc, dphi)
            dvdy[i_time, i_high, :, i_lon] = derivation.fft_diff(v[i_time, i_high, :, i_lon] * wc, dphi)




for kh in kh_list:
    for ilambdaz in fre:
        gome = ilambdaz
        inum = 0
        axs= []
        axs1 = []
        axs2 = []
        axs3 = []
        axs4 = []
        axs5 = []
        axs6 = []
        for k, l, lonM, latM in zip(k_list, l_list, lonMlist, latMlist):
            k = k * 1.e-3/np.float64(kh)
            l = l * 1.e-3/np.float64(kh)
            lonMs = lonM
            latMs = latM
            if (lonMs == lonMlist[0]) and (latMs == latMlist[0]) :
                legend1 = [str(lonMs)+' E, ' + str(latMs) + ' N' ]
            timeM = her + mn/60
            zM = zini
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

            dudx1, dudy1, dvdx1, dvdy1, u1, v1, w1, NF1, H1, Cs2M = lunge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time1, z, lonM, latM, timeM, zM, dlat, dlon, Cs2 = Cs2)
            f = O2 * np.sin(latM/180*np.pi)
            ome = gome - (u1 * k1 + v1 * l1)
            print(lonM, latM)

            i = 0
            uM = np.NAN
            vM = np.NAN
            wM = np.NAN
            NFM = np.NAN
            HM = np.NAN
            Cs2M = np.NAN
            omeM = np.NAN
            dz = 0.1
            try:        
                while dz:
                    dy, dx, dz, dk, dl, u1, v1, w1, NF1, ome1, H1, Cs2M1 = lunge_kutta4.main_lunge(dudx, dvdx, dudy, dvdy, u, v, wv, NF, H, gome, k1, l1, lon, lat, time1, z, lonM1, latM1, timeM1, zM1, phi, ram, dt, dlat, dlon, Cs2 = Cs2)
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
                        Cs2M = Cs2M1
                    else:
                        uM = np.append(uM, u1)
                        vM = np.append(vM, v1)
                        wM = np.append(wM, w1)
                        NFM = np.append(NFM, NF1)
                        HM = np.append(HM, H1)
                        omeM = np.append(omeM, ome1)     
                        Cs2M = np.append(Cs2M, Cs2M1)   
                    i = i + 1
                        #print(i)
            except:
                #save or plot output here
                inum += 1
                print('end')

print('end')
                