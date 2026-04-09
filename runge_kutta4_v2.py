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
## Version: 3.1.0
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2026/04/09
## The latest bugfix information can be found in "read_me.txt".
##################################################
import numpy as np
from meteo_para_v3 import meteo_para
global r, O2, const
r = 6.3781 * 1.e6 #[m] Mean radius of the earth
O2 = 0.00014584231#7.2921159 * 1e-5 * 2 #rad/s
const = 29.26 # m/K
g = 9.80665 #[m/s^2]
class runge_kutta4:
    def main_runge(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, gome, k, l, lon, lat, time, z, lonM, latM, timeM, zM, phi, ram, dt, dlat, dlon, Cs2 = None):
        import numpy as np
        #-------first step-----------------
        if np.isnan(zM):
            return (np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        f1, b1, dudx1, dvdx1, dudy1, dvdy1, u1, v1, w1, NF1, H1, Cs21 = runge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM, latM, timeM, zM, dlat, dlon, Cs2 = Cs2)
        if np.isnan(zM):
            return np.nan, np.nan, np.nan, np.nan, np.nan, u1, v1, w1, NF1, np.nan, H1, Cs21
        step = 1
        zM2, lonM2, latM2, timeM2, k2, l2, ug1, vg1, wg1, ome1, dk1 ,dl1 = runge_kutta4.runge_step(H1, gome, k, k, u1, l, l, v1, latM, lonM, NF1, w1, dudx1, dvdx1, dudy1, dvdy1, zM, timeM, step, dt, f1, b1, Cs2 = Cs21)
        if np.isnan(zM2):
            return zM2, zM2, zM2, zM2, zM2, u1, v1, w1, NF1, ome1, H1, Cs21
        f2, b2, dudx2, dvdx2, dudy2, dvdy2, u2, v2, w2, NF2, H2, Cs22 = runge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM2, latM2, timeM2, zM2, dlat, dlon, Cs2 = Cs2)

        step = 2
        zM3, lonM3, latM3, timeM3, k3, l3, ug2, vg2, wg2, ome2, dk2 ,dl2 = runge_kutta4.runge_step(H2, gome, k, k2, u2, l, l2, v2, latM, lonM, NF2, w2, dudx2, dvdx2, dudy2, dvdy2, zM, timeM, step, dt, f2, b2, Cs2 = Cs22)
        if np.isnan(zM3):
            return zM3, zM3, zM3, zM3, zM3, u1, v1, w1, NF1, ome1, H1, Cs21
        f3, b3, dudx3, dvdx3, dudy3, dvdy3, u3, v3, w3, NF3, H3, Cs23 = runge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM3, latM3, timeM3, zM3, dlat, dlon, Cs2 = Cs2)

        step = 3
        zM4, lonM4, latM4, timeM4, k4, l4, ug3, vg3, wg3, ome3, dk3 ,dl3 = runge_kutta4.runge_step(H3, gome, k, k3, u3, l, l3, v3, latM, lonM, NF3, w3, dudx3, dvdx3, dudy3, dvdy3, zM, timeM, step, dt, f3, b3, Cs2 = Cs23)
        if np.isnan(zM4):
            return zM4, zM4, zM4, zM4, zM4, u1, v1, w1, NF1, ome1, H1, Cs21
        f4, b4, dudx4, dvdx4, dudy4, dvdy4, u4, v4, w4, NF4, H4, Cs24 = runge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM4, latM4, timeM4, zM4, dlat, dlon, Cs2 = Cs2)
        step = 4
        ug4, vg4, wg4, ome4, dk4 ,dl4 = runge_kutta4.runge_step(H4, gome, k, k4, u4, l, l4, v4, latM, lonM, NF4, w4, dudx4, dvdx4, dudy4, dvdy4, zM, timeM, step, dt, f4, b4, Cs2 = Cs24)
        dy = 1./6. * (vg1 + vg2 * 2. + vg3 * 2. + vg4) * (dt)/r * 180 /np.pi
        dx = 1./6. * (ug1 + ug2 * 2. + ug3 * 2. + ug4) * (dt)/(r * np.abs(np.cos((latM + dy * 0.5)/180 * np.pi))) * 180 /np.pi
        dz = 1./6. * (wg1 + wg2 * 2. + wg3 * 2. + wg4) * (dt)
        dk = 1./6. * (dk1 + dk2 * 2. + dk3 * 2. + dk4) * (dt)
        dl = 1./6. * (dl1 + dl2 * 2. + dl3 * 2. + dl4) * (dt)
            
        return dy, dx, dz, dk, dl, u1, v1, w1, NF1, ome1, H1, Cs21

    
    def data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM, latM, timeM, zM, dlat, dlon, Cs2 = None):
        import numpy as np
        from scipy.interpolate import interpn  
        zmin = np.nanmax(np.where(z[np.argmin(np.abs(np.round(time - np.min(timeM)))), :, int(np.round((90 - latM)/dlat)), int(np.round(lonM/dlon))] <= zM)) - 1
        zmax = np.nanmin(np.where(z[np.argmin(np.abs(np.round(time - np.min(timeM)))), :, int(np.round((90 - latM)/dlat)), int(np.round(lonM/dlon))] > zM)) + 1

        f = O2 * np.sin(latM/180*np.pi)
        b = O2 * np.cos(latM/180*np.pi)/r
        
        num = zmax - zmin + 1 
        dudxM1 = np.zeros(num)
        dvdxM1 = np.copy(dudxM1)
        dudyM1 = np.copy(dudxM1)
        dvdyM1 = np.copy(dudxM1)
        uM1 = np.copy(dudxM1)
        vM1 = np.copy(dudxM1)
        wM1 = np.copy(dudxM1)
        NFM1 = np.copy(dudxM1)
        HM1 = np.copy(dudxM1)
        zM1 = np.copy(dudxM1)
        if Cs2 is not None:
            Cs2M1 = np.copy(dudxM1)
         
        for i_high in range(zmin,zmax+1):
            dudxM1[i_high - zmin] = interpn([time, 90 + lat, lon], dudx[:,i_high,:,:], [timeM, 90 + latM, lonM])
            dvdxM1[i_high - zmin] = interpn([time, 90 + lat, lon], dvdx[:,i_high,:,:], [timeM, 90 + latM, lonM])
            dudyM1[i_high - zmin] = interpn([time, 90 + lat, lon], dudy[:,i_high,:,:], [timeM, 90 + latM, lonM])
            dvdyM1[i_high - zmin] = interpn([time, 90 + lat, lon], dvdy[:,i_high,:,:], [timeM, 90 + latM, lonM])
            uM1[i_high - zmin] = interpn([time, 90 + lat, lon], u[:,i_high,:,:], [timeM, 90 + latM, lonM])
            vM1[i_high - zmin] = interpn([time, 90 + lat, lon], v[:,i_high,:,:], [timeM, 90 + latM, lonM])
            wM1[i_high - zmin] = interpn([time, 90 + lat, lon], w[:,i_high,:,:], [timeM, 90 + latM, lonM])
            NFM1[i_high - zmin] = interpn([time, 90 + lat, lon], NF[:,i_high,:,:], [timeM, 90 + latM, lonM])
            HM1[i_high - zmin] = interpn([time, 90 + lat, lon], H[:,i_high,:,:], [timeM, 90 + latM, lonM])
            zM1[i_high - zmin] = interpn([time, 90 + lat, lon], z[:,i_high,:,:], [timeM, 90 + latM, lonM])
            if Cs2 is not None:
                Cs2M1[i_high - zmin] = interpn([time, 90 + lat, lon], Cs2[:,i_high,:,:], [timeM, 90 + latM, lonM])

          
        dudxM = np.interp(zM, zM1, dudxM1) 
        dvdxM = np.interp(zM, zM1,  dvdxM1) 
        dudyM = np.interp(zM,  zM1,  dudyM1) 
        dvdyM = np.interp(zM,  zM1,  dvdyM1)   
        uM = np.interp(zM,  zM1,  uM1)  
        vM = np.interp(zM,  zM1,  vM1)
        wM = np.interp(zM,  zM1,  wM1) 
        NFM = np.interp(zM, zM1,  NFM1) 
        HM = np.interp(zM,  zM1,  HM1)
        if Cs2 is not None:
            Cs2M = np.interp(zM,  zM1, Cs2M1)
        else:
            Cs2M = None
        return f, b, dudxM, dvdxM, dudyM, dvdyM, uM, vM, wM, NFM, HM, Cs2M

   
    def runge_step(H, gome, k, kn, u, l, ln, v, latM, lonM, NF, w, dudx, dvdx, dudy, dvdy, zM, timeM, step, dt, f, b, Cs2 = None):
        dt_2 = dt * 0.5
        dt7200 = dt/7200.
        ome = np.abs(gome - (kn * u + ln * v))
        m = -np.sqrt(meteo_para.cal_dis_ral( kn, ln, ome, f, NF, H))
        dk = -(kn * dudx + ln * dvdx)
        dl = -(kn * dudy + ln * dvdy + b * f/ome)
        ug, vg, wg = meteo_para.group_velocity( kn, ln, m, H, NF, ome, w, f, u, v, cs2=Cs2)
        #-------intermediate RK step---------
        if step == 1 or step == 2:
            zM2 = zM + dt_2 * wg
            lonM2 = lonM + np.degrees(dt_2 * ug / (r * abs(np.cos(np.radians(latM) + dt_2 * vg / r))))
            latM2 = latM + np.degrees(dt_2 * vg /r)
            timeM2 = timeM + dt7200
            k2 = k + dk * dt_2
            l2 = l + dl * dt_2
            return zM2, lonM2, latM2, timeM2, k2, l2, ug, vg, wg, ome, dk ,dl
        elif step == 3:
            zM2 = zM + dt * wg
            lonM2 = lonM + np.degrees(dt * ug / (r * abs(np.cos(np.radians(latM) + dt_2 * vg / r))))
            latM2 = latM + np.degrees(dt * vg /r)
            timeM2 = timeM + dt7200 * 2
            k2 = k + dk * dt
            l2 = l + dl * dt
            return zM2, lonM2, latM2, timeM2, k2, l2, ug, vg, wg, ome, dk ,dl
        elif step == 4:
            if np.isnan(wg):
                print("dame") 
            return ug, vg, wg, ome, dk ,dl
        
        
    def instability(Tin, NF, uM, vM, zM, HM, latM, lonM, timeM, omeM, kM, lM, mM, rho, T, z, dlat, dlon, time, lat, lon ):
        import numpy as np
        from scipy.interpolate import interpn  
       
        du = np.gradient(uM, zM)
        dv = np.gradient(vM, zM)
        rhoM = np.copy(uM)
        TM = np.copy(uM)
        
        
        Ri = np.zeros_like(NF)
        RiN = np.zeros_like(NF)
        N_full = np.copy(uM)

        NSH1 = len(du)
        Et = 0

        for i in range(len(rhoM)):
            zmin = np.nanmax(np.where(z[int(np.round(timeM[i] - np.min(timeM))), :, int(np.round(latM[i]/dlat)), int(np.round(lonM[i]/dlon))] <= zM[i]))
            try:
                zmax = np.nanmin(np.where(z[int(np.round(timeM[i]- np.min(timeM))), :, int(np.round(latM[i]/dlat)), int(np.round(lonM[i]/dlon))] > zM[i])) 
                rhoM1 = np.zeros(2)    
                TM1 = np.zeros(2)      
                zM1 = np.zeros(2)         
                for i_high in [zmin, zmax]:
                    rhoM1[i_high - zmin] = interpn([time, 90 - lat, lon], rho[:,i_high,:,:], [timeM[i], 90 - latM[i], lonM[i]])      
                    TM1[i_high - zmin] = interpn([time, 90 - lat, lon], T[:,i_high,:,:], [timeM[i], 90 - latM[i], lonM[i]])         
                    zM1[i_high - zmin] = interpn([time, 90 - lat, lon], z[:,i_high,:,:], [timeM[i], 90 - latM[i], lonM[i]])
                rhoM[i] = np.exp(np.interp(zM[i], zM1, np.log(rhoM1)))
                TM[i] = np.interp(zM[i], zM1, TM1)
            except:
                print("exceeding the upper boundary of the model")
                rhoM[i] = np.nan
                TM[i] = np.nan
        f = O2 * np.sin(np.radians(latM))
        mM = -np.sqrt(meteo_para.cal_dis_ral( kM, lM, omeM, f, NF, HM))

        for SH1 in range(NSH1):
            if SH1 != 0:
                Et *= (np.sqrt(rhoM[SH1-1]) / np.sqrt(rhoM[SH1]))
                Ep = Et * (omeM[SH1]**2 - f[SH1]**2) / (2. * omeM[SH1]**2)
                Ek = Et * ((omeM[SH1]**2 + f[SH1]**2) / (2. * omeM[SH1]**2))
                Tamp = np.sqrt(Ep * 4. * NF[SH1] / g**2)

            else:
                Tamp = Tin/TM[SH1]
                Ep = 0.25 * g**2 / NF[SH1] * (Tamp)**2
                Ek = (omeM[SH1]**2 + f[SH1]**2) / (omeM[SH1]**2 - f[SH1]**2) * Ep
                Et = Ep + Ek

            upara = np.sqrt(4. * Ek * (f[SH1]**2 / (omeM[SH1]**2 + f[SH1]**2)))
            uperp = upara * f[SH1] / omeM[SH1]

            N_full[SH1] = NF[SH1] - g * np.abs(mM[SH1]) * Tamp

            angles = np.radians(np.arange(361))
            S = np.sin(angles)
            C = np.cos(angles)

            denom = np.sqrt(kM[SH1]**2 + lM[SH1]**2)
            uprim = upara * kM[SH1] / denom * C - uperp * lM[SH1] / denom * S
            vprim = upara * lM[SH1] / denom * C + uperp * kM[SH1] / denom * S
            tprim = -Tamp * np.sin(angles)

            duprim = mM[SH1] * uprim
            dvprim = mM[SH1] * vprim
            dNprim = mM[SH1] * tprim * g

            du2_total = (du[SH1] + duprim)**2
            dv2_total = (dv[SH1] + dvprim)**2
            dNprim_total = NF[SH1] + dNprim

            Ri[SH1] = np.min(dNprim_total / (du2_total + dv2_total))
            RiN[SH1] = NF[SH1] / (du[SH1]**2 + dv[SH1]**2)

        op = np.where(Ri <= 0.25)[0]

        if op.size > 0:
            op_min = op.min()
            print(zM[op_min])
            print(op_min)
        else:
            print("No Ri <= 0.25 found.")
            
        return N_full, Ri, mM
