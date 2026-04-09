#Kogure updated kernel_box on 2022 August 31.
#add butterworth_kernel
class kernel_box:
    
    def butterworth_kernel(k,l = [], orderL = [], orderH = [], CUTL= [], CUTH = [], highps = [], lowps = [], bandps = [], dim =1 ):
    #http://www.ic.is.tohoku.ac.jp/~swk/lecture/yaruodsp/butterworth.html
    #http://www.ic.is.tohoku.ac.jp/~swk/lecture/yaruodsp/dfdesign.html#link:Sec:iir_indirect
    #https://www.originlab.com/doc/Origin-Help/2DFFT-Filter-Algorithm
        import numpy as np
        if highps: 
            if dim == 1:
                H = 1./np.sqrt(1. + (CUTH/k)**(2. * orderH))
            if dim == 2:
                H = np.zeros([len(k), len(l)])
                for i_l in range(1, len(l)):
                    H[:,i_l] = 1./(np.sqrt(1. + (CUTH/np.sqrt(k**2 + l[i_l]**2))**(2. * orderH)))
                H[(k==0),(l==0)] = 0 
            return H
            
        if bandps:
            if dim == 1:
                H = 1./(np.sqrt(1. + (k/CUTL)**(2. * orderL))) * 1./(np.sqrt(1. + (CUTH/k))**(2. * orderH))
            if dim == 2:
                H = np.zeros([len(k), len(l)])
                i_l = 0
                H[i_l,i_l] = 0
                for i_l in range(1, len(l)):
                    H[:,i_l] = 1./(np.sqrt(1. + (CUTL/np.sqrt(k**2 + l[i_l]**2))**(2. * orderL))) * 1./(np.sqrt(1. + (np.sqrt(k**2 + l[i_l]**2)/CUTH)**(2. * orderH)))
                H[(k==0),(l==0)] = 0 
            return H

        if lowps:
            if dim == 1:
                H = 1./(np.sqrt(1. + (k/CUTL)**(2. * orderL)))
            if dim == 2:
                H = np.zeros([len(k), len(l)])
                for i_l in range(len(l)):
                    H[:,i_l] = 1./(np.sqrt(1. + (np.sqrt(k**2 + l[i_l]**2)/CUTL)**(2. * orderL)))
                    H[(k==0),(l==0)] = 0 
            return H
    
    def cosine_roll(n, parcent):
        import numpy as np
        w = np.zeros(n) + 1
        w[0:int(n*parcent)] = np.cos(np.pi/2 - np.pi/2 * np.arange(int(n*parcent))/(int(n*parcent)-1))
        w[int(n - n*parcent)+1:n] = np.flip(np.cos(np.pi/2 - np.pi/2 * np.arange(int(n*parcent))/(int(n*parcent)-1)))
        return w
    
    def hanning(nx, dim, ny = []):
    #http://www.ic.is.tohoku.ac.jp/~swk/lecture/yaruodsp/win.html
    #https://ja.wikipedia.org/wiki/%E7%AA%93%E9%96%A2%E6%95%B0
        import numpy as np
        if dim == 1:
            w = 0.50 - 0.5 * np.cos(2 * np.pi * np.arange(0, nx)/nx)
            return w
        if dim == 2:
           # w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.sqrt((nx - np.transpose(np.tile(np.arange(0, nx), (ny, 1))))**2 + (ny - np.tile(np.arange(0, ny), (nx, 1)))**2)/np.sqrt(nx**2 + ny**2)))
            w = 0.5 + 0.5 * np.cos(2 * np.pi * np.sqrt(((nx/2 - np.transpose(np.tile(np.arange(0, nx), (ny, 1))))**2 + (ny/2 -np.tile(np.arange(0, ny), (nx, 1)))**2)/(nx**2 + ny**2)))
            return w 
    
    def cal_order(CUT, CUTC, A, highps = [], lowps = []):
        import numpy as np
        if highps:
            N = np.ceil(np.log(1/A**2 - 1)/(2 * np.log(CUT/CUTC)))
        if lowps:
            N = np.ceil(np.log(1/A**2 - 1)/(2 * np.log(CUTC/CUT)))
        return N
        #http://www.ic.is.tohoku.ac.jp/~swk/lecture/yaruodsp/butterworth.html#SECTION002020000000000000000
        
    def gauss(WINDOW, HWHM ):
        import numpy as np
        sigma = HWHM/np.sqrt(2 * np.log(2))
        Gfun = np.exp(-((np.arange(WINDOW) + 0.5) - WINDOW * 0.5) ** 2/(2 * sigma ** 2))
        return Gfun