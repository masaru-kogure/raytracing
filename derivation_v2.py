##################################################
## Description
## This program calculates numerical derivatives used by the ray-tracing model. 
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
## Version: 2.1.1
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2026/04/09
## Bug fixed for FFT difference
##################################################
import numpy as np
class derivation:
    
    def fft_diff_axis(vari, dx, axis, idel_filter=None):
        wave = np.fft.fftfreq(vari.shape[axis], d=dx) * 2 * np.pi
        if idel_filter is not None:
            wave = wave.copy()
            wave[np.abs(wave) >= idel_filter] = 0
        wave_shape = [1] * vari.ndim
        wave_shape[axis] = wave.size
        wave = wave.reshape(wave_shape)
        return np.real(np.fft.ifft(np.fft.fft(vari, axis=axis) * 1j * wave, axis=axis))
    
    def central_diff(vari, dx):
        dvari = np.convolve(vari, [-1, 0, 1], mode = 'same')/(2.*dx)
        dvari[0] = (vari[1] - vari[0])/(dx) 
        dvari[-1] = (vari[-2] - vari[-1])/(dx)  
        return dvari

    def uneven_central_diff(vari, x):
        dvari = np.convolve(vari, [-1, 0, 1], mode = 'same')/(np.roll(x, 1) - np.roll(x, -1))
        dvari[0] = (vari[1] - vari[0])/(x[1] - x[0])
        dvari[-1] = (vari[-1] - vari[-2])/(x[-1] - x[-2]) 
        return dvari
    
    def interpol_diff(x, y):
        x01 = np.roll(x, -1) - x
        x01[0] = x[0] - x[1]
        x01[-1] = x[-2] - x[-1]
        x02 = np.roll(x, -1) - np.roll(x, 1)
        x02[0] = x[0] - x[2]
        x02[-1] = x[-1-2] - x[-1]
        x12 = x - np.roll(x, 1)
        x12[0] = x[1] - x[2]
        x12[-1] = x[-1-1] - x[-1]
        dy = np.roll(y, -1) * x12 / (x01 * x02) + y * (1./x12 - 1./x01) - np.roll(y, 1) * x01 / (x02 * x12)
        dy[0] = y[0] * (x01[0] + x02[0]) / (x01[0] * x02[0]) - y[1] * x02[0]/(x01[0] * x12[0]) + y[2] * x01[0] / (x02[0] * x12[0])
        dy[-1] = -y[-1-2] * x12[-1] / (x01[-1] * x02[-1]) + y[-1-1] * x02[-1]/(x01[-1] * x12[-1]) - y[-1] * (x02[-1] + x12[-1]) / (x02[-1] * x12[-1]) 
        return dy
        #https://www.l3harrisgeospatial.com/docs/deriv.html

    def interpol_diff2(x, y):
        x01 = np.roll(x, -1) - x
        x01[0] = x[0] - x[1]
        x01[-1] = x[-2] - x[-1]
        x02 = np.roll(x, -1) - np.roll(x, 1)
        x02[0] = x[0] - x[2]
        x02[-1] = x[-1-2] - x[-1]
        x12 = x - np.roll(x, 1)
        x12[0] = x[1] - x[2]
        x12[-1] = x[-1-1] - x[-1]
        dy2 = (np.roll(y, -1)/ (x01 * x02) + np.roll(y, 1) / (x02 * x12) - y * 1./(x12 * x01)) * 2
        dy2[0] = (y[0]/ (x01[0] * x02[0]) + y[1] / (x02[0] * x12[0]) - y[2] * 1./(x12[0] * x01[0])) * 2
        dy2[-1] = (y[-3]/ (x01[-1] * x02[-1]) + y[-2] / (x02[-1] * x12[-1]) - y[-1] * 1./(x12[-1] * x01[-1])) * 2
        return dy2
        #https://www.l3harrisgeospatial.com/docs/deriv.html
    
    def fft_diff(vari, dx, w=1, idel_filter= []):
        # Spectral first derivative. np.fft.fftfreq handles both even and odd lengths.
        wave = np.fft.fftfreq(len(vari),d=dx) * 2 * np.pi
        if idel_filter:
            nd = np.abs(wave) >= idel_filter
            wave[nd] = 0
        dvari = np.real(np.fft.ifft(np.fft.fft(vari) * 1j * wave * w))
        return dvari
    
    def fft_diff2(vari, dx, w=1, idel_filter= []):
        # Spectral second derivative. np.fft.fftfreq handles both even and odd lengths.
        wave = np.fft.fftfreq(len(vari),d=dx) * 2 * np.pi
        if idel_filter:
            nd = np.abs(wave) >= idel_filter
            wave[nd] = 0
        dvari = np.real(np.fft.ifft(np.fft.fft(vari) * (-1) * (wave)**2 * w))
        return dvari
    
    def central_diff2(vari, dx):
        dvari = np.convolve(vari, [1, -2, 1], mode = 'same')/(dx**2)
        dvari[0] = (vari[2] - 2*vari[1] + vari[0])/(dx**2) 
        dvari[-1] = (vari[-3] - 2*vari[-2] + vari[-1])/(dx**2)  
        return dvari
