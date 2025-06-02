class derivation:
    
    def central_diff(vari, dx):
        import numpy as np
        dvari = np.convolve(vari, [-1, 0, 1], mode = 'same')/(2.*dx)
        dvari[0] = (vari[1] - vari[0])/(dx) 
        dvari[-1] = (vari[-2] - vari[-1])/(dx)  
        return dvari

    def uneven_central_diff(vari, x):
        import numpy as np
        dvari = np.convolve(vari, [-1, 0, 1], mode = 'same')/(np.roll(x, 1) - np.roll(x, -1))
        dvari[0] = (vari[1] - vari[0])/(x[1] - x[0])
        dvari[-1] = (vari[-1] - vari[-2])/(x[-1] - x[-2]) 
        return dvari
    
    def interpol_diff(x, y):
        import numpy as np
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
    
    def fft_diff(vari, dx, w=1, idel_filter= []):
        #window function
        # https://numpy.org/doc/stable/reference/routines.window.html
        import numpy as np
        #X = np.arange(int((len(vari) - 1)/2)) + 1
        #is_N_even = (np.mod(len(vari),2) == 0)
        #if is_N_even:
        #    wave = np.hstack([0, X, len(vari)/2, -(len(vari)/2 + 1) + X])/(len(vari) * dx)
        #else:
        #    wave = np.hstack([0, X, -(len(vari)/2) + X])/(len(vari) * dx) 
        wave = np.fft.fftfreq(len(vari),d=dx) * 2 * np.pi
        if idel_filter:
            nd = wave >= idel_filter
            wave[nd] = 0
        dvari = np.real(np.fft.ifft(np.fft.fft(vari) * 1j * wave * w))
        return dvari
    
        #butterworth filter
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html#scipy.signal.buttord