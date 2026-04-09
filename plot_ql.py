##################################################
## Description
## This program saves ray-tracing results and makes quick-look diagnostic plots. 
## This code is maintained for Python 3.
##################################################
## Terms and Conditions of Use  
## This program is free to use for academic, non-commercial purposes. 
## Modification of the code is not recommended; any modifications are made at your own risk. 
## We strongly encourage users to contact us for discussion before using the result of this software in publications, 
## to prevent misuse or misinterpretation of the output.
## The developers and their affiliated organizations are not responsible for any damages arising from use of the software.
##################################################
## Author: Masaru Kogure
## Version: 1.0.0
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2026/04/09
##################################################
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
class plot_ql:
    def plot_ql_ray(save_dir, case_id, lonM, latM, zM, timeM, uM, vM, NFM, Ri, mmo1, dirM):
        # Make one compact diagnostic figure for a completed ray.
        save_path = save_dir / f"{case_id}_ql.png"
        z_km = zM * 1.e-3

        fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        fig.suptitle(case_id)

        axs[0, 0].plot(lonM, latM, "-k", lw=1.5)
        axs[0, 0].scatter(lonM[0], latM[0], c="tab:red", s=35, label="start")
        axs[0, 0].scatter(lonM[-1], latM[-1], c="tab:blue", s=35, label="end")
        axs[0, 0].set_xlabel("Longitude [deg]")
        axs[0, 0].set_ylabel("Latitude [deg]")
        axs[0, 0].legend(loc="best")

        axs[0, 1].plot(timeM, z_km, "-k", lw=1.5)
        axs[0, 1].set_xlabel("Time [UT hour]")
        axs[0, 1].set_ylabel("Height [km]")

        axs[0, 2].plot(uM, z_km, label="u")
        axs[0, 2].plot(vM, z_km, label="v")
        axs[0, 2].set_xlabel("Wind [m/s]")
        axs[0, 2].set_ylabel("Height [km]")
        axs[0, 2].legend(loc="best")

        axs[1, 0].plot(NFM * 1.e4, z_km, "-k")
        axs[1, 0].set_xlabel("N^2 [1e-4 s^-2]")
        axs[1, 0].set_ylabel("Height [km]")

        axs[1, 1].plot(mmo1, z_km, "-k")
        axs[1, 1].set_xlabel("Vertical wavelength [km]")
        axs[1, 1].set_ylabel("Height [km]")

        axs[1, 2].plot(Ri, z_km, "-k", label="Ri")
        axs[1, 2].axvline(0.25, color="tab:red", ls="--", lw=1, label="0.25")
        axs[1, 2].set_xlabel("Richardson number")
        axs[1, 2].set_ylabel("Height [km]")
        axs[1, 2].legend(loc="best")

        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
        axs[0, 0].set_title(f"direction start={dirM[0]:.1f} deg")

        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path


    def save_ray_npz(save_dir, case_id, **ray):
        # Store numeric ray products in a compressed file for later analysis.
        save_path = save_dir / f"{case_id}.npz"
        np.savez_compressed(save_path, **ray)
        return save_path
    
    def ray_case_id(date, kh, lambdaz, lon0, lat0):
        label = f"{date}_kh{kh}_lz{lambdaz:g}_lon{lon0:.1f}_lat{lat0:.1f}"
        return label.replace("-", "m").replace(".", "p")
