import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda
from skcuda.fft import *
import numpy as np



N = 4
#X = np.asarray(np.random.rand(N), np.float32)
X = np.asarray([1,1,1,1], np.float32)
x_gpu = gpuarray.to_gpu(X)
xf_gpu = gpuarray.empty(3, np.complex64)

plan = skcuda.cufft.cufftPlan1d(nx=4, fft_type=skcuda.cufft.CUFFT_R2C,batch=1)

skcuda.cufft.cufftExecR2C(plan, x_gpu, xf_gpu)
'''
#cufftMakePlanMany(plan, rank=1, )



fft(x_gpu, xf_gpu, plan)
print xf_gpu.get()
print np.fft.fft(X)




'''

