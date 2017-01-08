import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda.fft import *
import numpy as np



N = 4
B = 2
X = np.asarray([1,1,1,1,1,1,1,1], np.float32)
x_gpu = gpuarray.to_gpu(X)
xf_gpu = gpuarray.empty((N/2+1)*B, np.complex64)

plan = Plan(shape=4, in_dtype=np.float32, out_dtype=np.complex64, batch=B, istride=1, idist=4, ostride=1, odist=3)

fft(x_gpu, xf_gpu, plan)
print xf_gpu.get()

'''

import time
s = time.time()
fft(x_gpu, xf_gpu, plan)
print xf_gpu.get().shape
print time.time() - s

import librosa
s = time.time()
xf = np.fft.fft(X[:N])
print xf_gpu.get()
print xf
print time.time() -s 

print np.allclose(xf[0:N/2+1], xf_gpu.get()[0:N/2+1], atol=1e-6)





'''
