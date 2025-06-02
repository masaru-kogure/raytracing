##################################################
## Discripition
## This program calculates meteorological parameters. 
## This code is developted in Python 3.8.19
##################################################
## Terms and Conditions of Use  
## This program is free to use for academic, non-commercial purposes. 
## Modification of the code is not recommended; any moddifications are made at your own risk. 
## We strongly encourage users to contact us for discussion before using the result of this software in publications, 
## to prevent misuse or misinterpretation of the output.
## The developers and their affiliated organizations are not responsible for any damages arising from use of the software.
##################################################
## Author: Masaru Kogure
## Version: 2.0.1
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2025/06/02
##################################################

from os import RTLD_DEEPBIND
import numpy as np
from derivation import derivation
        #-------------constant----------------------
        #Cp = 1004. ;[J/(K*kg)]
        #R = 287. ;[J/(K*kg)]
        #An introduction to dynamic meteorogy p.491
        #ps = 1000. ;[hpa]
        #-------------constant----------------------
        #np.transpose(P.reshape(37,1,1,1), [1,0,2,3]
global g0, r, R0
g0 = 9.80665 #[m/s^2]
r = 6.3781 * 1.e6 #[m] Mean radius of the earth  
R0 = 8.3134 * 1.e3 #[J/(K*mol)] gas constant
OMEGA = 2 * 7.2921 * 1.e-5 #rad/s
class meteo_para:
    def poten_temp(t, P, Cp = 1004., R = 287.07):    
        P_T =  t * (1000./P)**(R/Cp) # = 0.28585657
        return P_T
    
    def cori_para(lat):
        f = OMEGA * 2 * np.sin(lat/180 * np.pi)
        return f
    
    def brunt_fre(z, P_T, g0 = 9.80665, dim = 0):        
        #NF1 = derivation.interpol_diff(z, P_T)
        if dim != 4:
            NF1 = np.gradient(P_T, z)
            NF = NF1 * (g0/P_T)
        else:
            ss = z.shape
            NF1 = z * np.nan
            for i_lon in range(ss[3]):
                for i_lat in range(ss[2]):
                    for i_time in range(ss[0]):
                        NF1[i_time, :, i_lat, i_lon] = np.gradient(P_T[i_time, :, i_lat, i_lon], z[i_time, :, i_lat, i_lon] )
            NF = NF1 * (g0/P_T)
        return NF
    
    def cal_dis_ral( k, l, omei, f, NFULL, H, cs2 = 0, m = 0):
        if m == 0:
            if cs2 == 0:
                m2 = (k**2 + l**2) * (NFULL - omei**2)/((omei**2 - f**2)) - 1/(4 * H**2)
            else:
                m2 = (k**2 + l**2) * (NFULL - omei**2)/((omei**2 - f**2)) - 1/(4 * H**2) + omei**2/cs2
            return m2
        else:
            ome2 = ((k**2 + l**2) * NFULL + f**2 * (m ** 2 + 1/(4 * H**2)))/(k**2 + l**2 + m ** 2 + 1/(4 * H**2))
            return ome2
    
    def group_velocity( k, l, m, H, NF, omei, w, f, u, v,):
        deno = (k**2 + l**2 + m**2 + 1/(H**2)) * omei
        CONS = (NF -omei**2)/deno
        ug = u + CONS * k
        vg = v + CONS * l 
        wg = (f**2 - omei**2) * m/deno + w
        return ug, vg, wg
    
    def density(P, T, R = 286.07):
        rho = P/(R *T)
        return rho
    
    def ver_wind(w, rho, g = 9.80665, z =[] ):
        if z != []:
            g = meteo_para.gravity_acce(z) 
        wver = -w/rho/g
        return wver
        
    def gas_const(N2 = 0.75527, O = 0, O2 = 0.23143, H2O = 0):#, CO2 = 0.0456):
        R = (N2/28.01 + O/18 + O2/32 + H2O/18.02 ) * R0#+ CO2/44.01)
        #p.13 一般気象学 (weigth ratio)
        return R
    
    def gravity_acce(z):
       g = g0 * (r ** 2 / (r + z)**2) 
       return g
   
    def scale_h(T, R = 287.07, g = 9.80665):
        H = T * R/g
        return H
    
    def dopper_h(colat, u, h = 694.02754, C0 = 465):
        hprim = h * (1 + u/(C0 * np.sin(colat))) ** 4
        return hprim
    
    def wavenumber2(NF2, g, H, hprim = 694.02754):
        kz2 = NF2/(g * hprim) - 1/(2 * H) ** 2
        return kz2
    
    def sound_wave_speed(t, Cp = 1004., R = 287.07):
        cs2 = Cp/(R * 5/2) * R * t
        return cs2
    
    def int_fre(ome, u, v, k, l):
        omei = ome - (u * k + v * l)
        return omei