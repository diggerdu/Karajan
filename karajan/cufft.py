import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda.fft import *
import numpy as np



N = 1024 * 60
X = np.asarray(np.random.rand(N), np.float32)
x_gpu = gpuarray.to_gpu(X)
xf_gpu = gpuarray.empty(N/2+1, np.complex64)

plan = Plan(shape=X.shape, in_dtype=np.float32, out_dtype=np.complex64, batch=60, idist=1024)



import time
s = time.time()
fft(x_gpu, xf_gpu, plan)
print xf_gpu.get().shape
print time.time() - s









