import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda.fft import *
import time
N = 44100 * 60 * 60 * 3
x = np.asarray(np.random.rand(N), np.float32)
s = time.time()
x_gpu = gpuarray.to_gpu(x)
xf_gpu = gpuarray.empty(N/2+1, np.complex64)
print time.time() - s
plan = Plan(x.shape, np.float32, np.complex64)
s= time.time()
fft(x_gpu, xf_gpu, plan)
print time.time() - s


import scipy.fftpack
s = time.time()
scipy.fftpack.fft(x)
print time.time() -s 
