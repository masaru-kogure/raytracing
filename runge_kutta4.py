##################################################
## Discripition
## This program calculates gravity wave paths by using 4th-order Runge-Kutta.
## It is translated from the IDL code presented in Kogure et al. (2018). 
## This code is developed in Python 3.8.19
##################################################
## Terms and Conditions of Use  
## This program is free to use for academic, non-commercial purposes. 
## Modification of the code is not recommended; any moddifications are made at your own risk. 
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
## Version: 3.0.1
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2025/06/02
##################################################
import numpy as np
from meteo_para_v2 import meteo_para
global r, O2, const
r = 6.3781 * 1.e6 #[m] Mean radius of the earth
O2 = 0.00014584231#7.2921159 * 1e-5 * 2 #rad/s
const = 29.26 # m/K
g = 9.80665 #[m/s^2]
class runge_kutta4:
    def main_runge(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, gome, k, l, lon, lat, time, z, lonM, latM, timeM, zM, phi, ram, dt, dlat, dlon):
        import numpy as np
        #-------firs step------------------
        if np.isnan(zM):
            print("dame")
        dudx1, dudy1, dvdx1, dvdy1, u1, v1, w1, NF1, H1 = lunge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM, latM, timeM, zM, dlat, dlon)
        if np.isnan(zM):
            print("dame")
        step = 1
        zM2, lonM2, latM2, timeM2, k2, l2, ug1, vg1, wg1, ome1, dk1 ,dl1 = lunge_kutta4.lunge_step(H1, gome, k, u1, l, v1, latM, lonM, NF1, w1, dudx1, dvdx1, dudy1, dvdy1, zM, timeM, step, dt)
        if np.isnan(zM2):
            print("dame") 
        dudx2, dudy2, dvdx2, dvdy2, u2, v2, w2, NF2, H2 = lunge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM2, latM2, timeM2, zM2, dlat, dlon)

        step = 2
        zM3, lonM3, latM3, timeM3, k3, l3, ug2, vg2, wg2, ome2, dk2 ,dl2 = lunge_kutta4.lunge_step(H2, gome, k2, u2, l2, v2, latM2, lonM2, NF2, w2, dudx2, dvdx2, dudy2, dvdy2, zM2, timeM2, step, dt)
        if np.isnan(zM3):
            print("dame")  
        dudx3, dudy3, dvdx3, dvdy3, u3, v3, w3, NF3, H3 = lunge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM3, latM3, timeM3, zM3, dlat, dlon)

        step = 3
        zM4, lonM4, latM4, timeM4, k4, l4, ug3, vg3, wg3, ome3, dk3 ,dl3 = lunge_kutta4.lunge_step(H3, gome, k3, u3, l3, v3, latM3, lonM3, NF3, w3, dudx3, dvdx3, dudy3, dvdy3, zM3, timeM3, step, dt)
        if np.isnan(zM4):
            print("dame")  
        dudx4, dudy4, dvdx4, dvdy4, u4, v4, w4, NF4, H4 = lunge_kutta4.data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM4, latM4, timeM4, zM4, dlat, dlon)
        step = 4
        ug4, vg4, wg4, ome4, dk4 ,dl4 = lunge_kutta4.lunge_step(H4, gome, k4, u4, l4, v4, latM4, lonM4, NF4, w4, dudx4, dvdx4, dudy4, dvdy4, zM4, timeM4, step, dt)
        dy = 1./6. * (vg1 + vg2 * 2. + vg3 * 2. + vg4) * (dt)/r * 180 /np.pi
        dx = 1./6. * (ug1 + ug2 * 2. + ug3 * 2. + ug4) * (dt)/r * np.abs(np.cos((latM + dy * 0.5)/180 * np.pi)) * 180 /np.pi
        dz = 1./6. * (wg1 + wg2 * 2. + wg3 * 2. + wg4) * (dt)
        dk = 1./6. * (dk1 + dk2 * 2. + dk3 * 2. + dk4) * (dt)
        dl = 1./6. * (dl1 + dl2 * 2. + dl3 * 2. + dl4) * (dt)
        if np.isnan(dz):
            print("dame")
        if dx > 1:
            print("yabaiii")
        return dy, dx, dz, dk, dl, u1, v1, w1, NF1, ome1, H1 

        
    def data_interpo(dudx, dvdx, dudy, dvdy, u, v, w, NF, H, lon, lat, time, z, lonM, latM, timeM, zM, dlat, dlon):
        import numpy as np
        from scipy.interpolate import interpn  
        zmin = np.nanmax(np.where(z[int(np.round(timeM - np.min(timeM))), :, int(np.round(latM/dlat)), int(np.round(lonM/dlon))] <= zM))
        zmax = np.nanmin(np.where(z[int(np.round(timeM- np.min(timeM))), :, int(np.round(latM/dlat)), int(np.round(lonM/dlon))] >= zM)) 
        dudxM1 = np.zeros(2)
        dvdxM1 = np.zeros(2)
        dudyM1 = np.zeros(2)
        dvdyM1 = np.zeros(2)  
        uM1 = np.zeros(2) 
        vM1 = np.zeros(2) 
        wM1 = np.zeros(2) 
        NFM1 = np.zeros(2) 
        HM1 = np.zeros(2)
        zM1 = np.zeros(2)  
         
        for i_high in [zmin, zmax]:
            dudxM1[i_high - zmin] = interpn([time, 90 - lat, lon], dudx[:,i_high,:,:], [timeM, 90 - latM, lonM])
            dvdxM1[i_high - zmin] = interpn([time, 90 - lat, lon], dvdx[:,i_high,:,:], [timeM, 90 - latM, lonM])
            dudyM1[i_high - zmin] = interpn([time, 90 - lat, lon], dudy[:,i_high,:,:], [timeM, 90 - latM, lonM])
            dvdyM1[i_high - zmin] = interpn([time, 90 - lat, lon], dvdy[:,i_high,:,:], [timeM, 90 - latM, lonM])
            uM1[i_high - zmin] = interpn([time, 90 - lat, lon], u[:,i_high,:,:], [timeM, 90 - latM, lonM])
            vM1[i_high - zmin] = interpn([time, 90 - lat, lon], v[:,i_high,:,:], [timeM, 90 - latM, lonM])
            wM1[i_high - zmin] = interpn([time, 90 - lat, lon], w[:,i_high,:,:], [timeM, 90 - latM, lonM])
            NFM1[i_high - zmin] = interpn([time, 90 - lat, lon], NF[:,i_high,:,:], [timeM, 90 - latM, lonM])
            HM1[i_high - zmin] = interpn([time, 90 - lat, lon], H[:,i_high,:,:], [timeM, 90 - latM, lonM])
            zM1[i_high - zmin] = interpn([time, 90 - lat, lon], z[:,i_high,:,:], [timeM, 90 - latM, lonM])

          
        dudxM = np.interp(zM, zM1, dudxM1) 
        dvdxM = np.interp(zM, zM1, dvdxM1) 
        dudyM = np.interp(zM, zM1, dudyM1) 
        dvdyM = np.interp(zM, zM1, dvdyM1)   
        uM = np.interp(zM, zM1, uM1)  
        vM = np.interp(zM, zM1, vM1) 
        wM = np.interp(zM, zM1, wM1) 
        NFM = np.interp(zM, zM1, NFM1) 
        HM = np.interp(zM, zM1, HM1)
        return dudxM, dvdxM, dudyM, dvdyM, uM, vM, wM, NFM, HM
   
    def runge_step(H, gome, k, u, l, v, latM, lonM, NF, w, dudx, dvdx, dudy, dvdy, zM, timeM, step, dt):

        dt_2 = dt * 0.5
        dt7200 = dt/7200.
        ome = np.abs(gome - (k * u + l * v))
        f = O2 * np.sin(latM/180*np.pi)
        b = O2 * np.cos(latM/180*np.pi)/r
        m = -np.sqrt(meteo_para.cal_dis_ral( k, l, ome, f, NF, H))
        dk = -(k * dudx + l * dvdx)
        dl = -(k * dudy + l * dvdy + b * f/ome)
        ug, vg, wg = meteo_para.group_velocity( k, l, m, H, NF, ome, w, f, u, v)
        #-------second step------------------
        if step == 1 or step == 2:
            zM2 = zM + dt_2 * wg
            lonM2 = lonM + dt_2 * ug / (r * abs(np.cos(latM + dt_2 * vg / r)))
            latM2 = latM + dt_2 * vg * vg/r
            timeM2 = timeM + dt7200
            k2 = k + dk * dt_2
            l2 = l + dl * dt_2
            return zM2, lonM2, latM2, timeM2, k2, l2, ug, vg, wg, ome, dk ,dl
        elif step == 3:
            zM2 = zM + dt_2 * wg
            lonM2 = lonM + dt_2 * ug / (r * abs(np.cos(latM + dt_2 * vg / r)))
            latM2 = latM + dt_2 * vg * vg/r
            timeM2 = timeM + dt7200
            k2 = k + dk * dt_2
            l2 = l + dl * dt_2
            return zM2, lonM2, latM2, timeM2, k2, l2, ug, vg, wg, ome, dk ,dl
        elif step == 4:
            if np.isnan(wg):
                print("dame") 
            return ug, vg, wg, ome, dk ,dl
        
        
    def instability(Tin, NF, uM, vM, zM, HM, latM, lonM, timeM, omeM, kM, lM, mM, rho, T, z, dlat, dlon, time, lat, lon ):
        import numpy as np
        from scipy.interpolate import interpn  
       
        du = np.gradient(uM, zM[0:-1])
        dv = np.gradient(vM, zM[0:-1])
        rhoM = np.copy(uM)
        TM = np.copy(uM)
        
        
        Ri = np.zeros_like(NF)
        RiN = np.zeros_like(NF)
        N_full = np.copy(uM)

        NSH1 = len(du)
        Et = 0

        for i in range(len(rhoM)):
            zmin = np.nanmax(np.where(z[int(np.round(timeM[i] - np.min(timeM))), :, int(np.round(latM[i]/dlat)), int(np.round(lonM[i]/dlon))] <= zM[i]))
            zmax = np.nanmin(np.where(z[int(np.round(timeM[i]- np.min(timeM))), :, int(np.round(latM[i]/dlat)), int(np.round(lonM[i]/dlon))] >= zM[i])) 
            rhoM1 = np.zeros(2)    
            TM1 = np.zeros(2)      
            zM1 = np.zeros(2)         
            for i_high in [zmin, zmax]:
                rhoM1[i_high - zmin] = interpn([time, 90 - lat, lon], rho[:,i_high,:,:], [timeM[i], 90 - latM[i], lonM[i]])      
                TM1[i_high - zmin] = interpn([time, 90 - lat, lon], T[:,i_high,:,:], [timeM[i], 90 - latM[i], lonM[i]])         
                zM1[i_high - zmin] = interpn([time, 90 - lat, lon], z[:,i_high,:,:], [timeM[i], 90 - latM[i], lonM[i]])
            rhoM[i] = np.exp(np.interp(zM[i], zM1, np.log(rhoM1)))
            TM[i] = np.exp(np.interp(zM[i], zM1, TM1))

        f = O2 * np.sin(np.radians(latM))
        mM = -np.sqrt(meteo_para.cal_dis_ral( kM[0:-1], lM[0:-1], omeM, f[0:-1], NF, HM))

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
