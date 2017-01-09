import pycuda.autoinit
x = gpuarray.to_gpu(np.asarray([1.0,2.0,3.0,4.0], np.float32))
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg


import time
s = time.time()
linalg.init()
print time.time() - s
x = gpuarray.to_gpu(np.asarray([1.0,2.0,3.0,4.0], np.float32))
y = gpuarray.to_gpu(np.asarray([1.0,2.0,3.0,4.0], np.float32))

z = linalg.multiply(x,y)
print z.get()
