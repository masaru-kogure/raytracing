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
from func_plot_v13 import plot_2D
from func_plot_v13 import plot_1D
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
#--------read JAWARA data--------------------------
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
#--------read JAWARA data--------------------------
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
figuresave = '/home/masarukogure/masarukogure/raytrance/'

for i_high in range(NERA[1]):
    for i_time in range(NERA[0]):
        for i_lat in range(NERA[2]):
            dram1 = dram * np.abs(np.cos(phi[i_lat]))
            #ord, Wn = signal.buttord(1/4/dphi, 1/2/dphi, 3, 40, fs = 1/dram1 )
            #b, a = signal.butter( ord, Wn, 'low', fs = 1/dram1)
            
            #X = np.arange((NERA[3] - 1)/2) + 1
            #is_N_even = (np.mod(NERA[3],2) == 0)
            #if is_N_even:
            #    wave = np.hstack([0, X, NERA[3]/2, -(NERA[3]/2 + 1) + X])/(dram1 * NERA[3])
            #else:
            #    wave = np.hstack([0, X, -(NERA[3]/2 + 1) + X])/(dram1 * NERA[3]) 
            #https://note.nkmk.me/python-numpy-concatenate-stack-block/
            #wb, h = signal.freqs(b, a, wave)
            #plt.plot(wave,wb)
            #plt.show()
            
            dudx[i_time, i_high, i_lat,:] = derivation.fft_diff(u[i_time, i_high, i_lat,:], dram1, idel_filter=1/(2 * dphi))
            dvdx[i_time, i_high, i_lat,:] = derivation.fft_diff(v[i_time, i_high, i_lat,:], dram1, idel_filter=1/(2 * dphi))
            #if i_lat == 300: 
            #    plt.plot(lon, dudx[i_time, i_high, i_lat,:])
            #    plt.show()
        for i_lon in range(NERA[3]):
            dudy[i_time, i_high, :, i_lon] = derivation.fft_diff(u[i_time, i_high, :, i_lon] * wc, dphi)
            dvdy[i_time, i_high, :, i_lon] = derivation.fft_diff(v[i_time, i_high, :, i_lon] * wc, dphi)
            #if i_lon == 300: 
            #    plt.plot(lon, dudy[i_time, i_high, :, i_lon])
            #    plt.show()


#kh = "469"
#k = 2 * np.pi * 0.0020576131687242765 * 1.e-3
#l = 2 * np.pi * 0.0005527915975677203 * 1.e-3
#kh = "285"
#k = 2 * np.pi * 0.003086419753086417 * 1.e-3
#l = 2 * np.pi * 0.0016583747927031475 * 1.e-3
#kh = "135"
#k = 2 * np.pi * 0.007201646090534979 * 1.e-3
#l = 2 * np.pi * 0.0016583747927031518 * 1.e-3
#kh = "185"
#k = 2 * np.pi * 0.0051440329218107065 * 1.e-3
#l = 2 * np.pi * 0.0016583747927031518 * 1.e-3
kh_list = ["170", "210"] 
k_list = 2 * np.pi *  np.array([ np.cos(np.pi/2), np.cos(np.pi/4), np.cos(0), np.cos(-np.pi/4)])
l_list = 2 * np.pi * np.array([ np.sin(np.pi/2), np.sin(np.pi/4), np.sin(0), np.sin(-np.pi/4)])

zini = 300 * 1e3
lonMlist = np.array([131, 133.8, 135, 133.5])
latMlist = np.array([37, 36., 33, 31.])
pltnum = np.array(len(lonMlist))
fre = 2 * np.pi/np.array([17 * 60])
op_min = 0
cc_min = 0
list = ["map", "direction", "u", "v", "N", "time", "m"]
for ilist in list:
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
                    dir = 90 - (np.arctan2(lM, kM)/(np.pi) * 180)
                    
                    if (lonMs != lonMlist[0]) or (latMs != latMlist[0]):
                        legend1.append(str(lonMs)+' E, ' + str(latMs) + ' N')
                    if ilist == "map":
                        axs = plot_1D.raytrace_multi_vari_1D_plot( lonM, latM, zM * 1.e-3, np.array(lonM[op_min]), np.array(latM[op_min]), np.array(zM[op_min])* 1.e-3, np.array(lonM[cc_min]), np.array(latM[cc_min]), np.array(zM[cc_min])* 1.e-3, pltnum, inum, figtitle = [str(np.int16(her))+ '_' +kh +'_map' + str(np.int32(zini * 1.e-3))], title = [kh, 'lon', 'lat'], figuresave = figuresave, range1 = [10, np.int32(zini * 1.e-3)], hori_range = [130, 160, 32, 50], size = [7,7], cmap ='gist_rainbow', axs= axs)
                    if ilist == "direction":
                        axs1 = plot_1D.ray_normal_multi_vari_1D_plot( dir, zM * 1.e-3, dir[op_min], zM[op_min] * 1.e-3, dir[cc_min], zM[cc_min] * 1.e-3, pltnum,inum, legend = legend1, figtitle = [str(np.int16(her)) + '_'+kh + '_direction ' + str(np.int32(zini * 1.e-3)) ], title = [kh + ' direction', 'Height [km]', 'angle'], figuresave = figuresave, axs = axs1,  tfsize= 28, fsize = 24, fticksize = 30, cir_size = 30, hori_range = [-180, 180, 10, np.int32(zini * 1.e-3)])
                    if ilist == "u":
                        axs2 = plot_1D.ray_normal_multi_vari_1D_plot( uM, zM[0:-1] * 1.e-3, uM[op_min], zM[op_min] * 1.e-3, uM[cc_min], zM[cc_min] * 1.e-3, pltnum,inum, legend = legend1, figtitle = [str(np.int16(her))+ '_' +kh + '_zonal_wind ' + str(np.int32(zini * 1.e-3))], title = [kh + ' zonal wind', 'Height [km]', 'm/s'], figuresave = figuresave, axs = axs2,  tfsize= 28, fsize = 24, fticksize = 30, cir_size = 30, hori_range= [-160,100,10, np.int32(zini * 1.e-3)])
                    if ilist == "v":
                        nd = ~np.isnan(vM)
                        axs3 = plot_1D.ray_normal_multi_vari_1D_plot( vM, zM[0:-1] * 1.e-3, vM[op_min], zM[op_min] * 1.e-3, vM[cc_min], zM[cc_min] * 1.e-3, pltnum,inum, legend = legend1, figtitle = [str(np.int16(her))+ '_' +kh + '_meridional_wind ' + str(np.int32(zini * 1.e-3))], title = [kh + ' meridional wind', 'Height [km]', 'm/s'], figuresave = figuresave, axs = axs3, tfsize= 28, fsize = 24, fticksize = 30, cir_size = 30, hori_range= [-100,100,10,np.int32(zini * 1.e-3)])
                    if ilist == "N":
                        nd = ~np.isnan(NFM)
                        axs4 = plot_1D.ray_normal_multi_vari_1D_plot( NFM * 1e4, zM[0:-1] * 1.e-3, NFM[op_min]* 1e4, zM[op_min] * 1.e-3, NFM[cc_min]* 1e4, zM[cc_min] * 1.e-3, pltnum,inum, legend = legend1, figtitle = [str(np.int16(her)) + '_' +kh + '_N ' + str(np.int32(zini * 1.e-3))], title = [kh  +  ' sqared N', 'Height [km]', r'${10^{-4}}$ ${\times}$ s$^{-2}$'], figuresave = figuresave, axs = axs4,  tfsize= 28, fsize = 24, fticksize = 30, cir_size = 30, hori_range= [1,20,10,np.int32(zini * 1.e-3)], xlog = 1)
                    if ilist == "time":
                        axs5 = plot_1D.ray_normal_multi_vari_1D_plot( timeM, zM * 1.e-3, timeM[op_min], zM[op_min] * 1.e-3, timeM[cc_min], zM[cc_min] * 1.e-3, pltnum,inum, legend = legend1, figtitle = [str(np.int16(her)) + '_' +kh + '_time ' + str(np.int32(zini * 1.e-3))] , title = [kh  + ' time', 'Height [km]', 'Time [UT]'], figuresave = figuresave, axs = axs5,  tfsize= 28, fsize = 24, fticksize = 30, cir_size = 30, hori_range= [0,15,10,np.int32(zini * 1.e-3)])
                    if ilist == "m":
                        m = np.sqrt(meteo_para.cal_dis_ral( kM[0:-1], lM[0:-1], omeM, O2 * np.sin(latM[0:-1]/180*np.pi), NFM, HM, cs2 = Cs2M))
                        axs6 = plot_1D.ray_normal_multi_vari_1D_plot( 2 * np.pi/m * 1.e-3, zM[0:-1] * 1.e-3, 2 * np.pi/omeM[op_min], zM[op_min] * 1.e-3, 2 * np.pi/omeM[cc_min], zM[cc_min] * 1.e-3, pltnum,inum, legend = legend1, figtitle = [str(np.int16(her)) + '_' + kh + '_vertical_wavelength ' + str(np.int32(zini * 1.e-3))], title = [kh  +  ' time', 'Height [km]', 'Vertical wavelenth [km]'], figuresave = figuresave, axs = axs6,  tfsize= 28, fsize = 24, fticksize = 30, cir_size = 30, hori_range= [0,np.int32(zini * 1.e-3),10,np.int32(zini * 1.e-3)])

#[1, 7, 30, 95, 100]
                    inum += 1
                    print('end')

print('end')
                    