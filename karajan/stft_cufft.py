import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda.fft import *
import skcuda.linalg as linalg
linalg.init()
import scipy, pylab
import numpy as np
import scipy.io.wavfile as wave
import stftmo
def stft(x, fftsize=1024, overlap=4):   
    x = np.asarray(x, np.float32)
    hop = fftsize / overlap
    w = np.asarray(scipy.hanning(fftsize), np.float32)
    unroll_x = np.asarray([x[i:i+fftsize] for i in range(0, x.shape[0]-fftsize, hop)], np.float32)
    plan = Plan(shape=fftsize, in_dtype=np.float32, out_dtype=np.complex64, batch=unroll_x.shape[0], idist=fftsize, odist=fftsize/2+1)
    w_gpu = gpuarray.to_gpu(np.tile(w, unroll_x.shape[0]))
    x_gpu = gpuarray.to_gpu(unroll_x.flatten())
    xf_gpu = gpuarray.empty((fftsize/2+1)*unroll_x.shape[0], np.complex64)
    fft(linalg.multiply(w_gpu,x_gpu), xf_gpu, plan)
    s = xf_gpu.get()
    return s
def istft(X, scale = 1, overlap=4):   
    fftsize=(X.shape[1]-1)*2
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    x = x * scale
    return x.astype(np.int16)

if __name__ == '__main__':
    FILE = "spring.wav"
    fs = 8000        # sampled at 8 kHz
    framesz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.025      # and hop size of 25 milliseconds.

    (rate, rawData) = wave.read(FILE)
    
    LEN = rawData.shape[0]
    rawData = np.repeat(rawData, 1) 
    import time
    s = time.time()
    spec = np.abs(stft(rawData, fftsize=1024))
    print time.time() - s,'s'
   
    import librosa
    s = time.time()
    spec_or = np.abs(stftmo.stft(rawData, fftsize=1024))
    print spec_or.shape
    print time.time() - s,'s'
    #print np.allclose(spec,spec_or.flatten(),atol=1e-1)
    print np.sum(spec)
    print sum(spec_or.flatten())
    print np.sum(np.abs(spec-spec_or.flatten()))
    '''    reCon = istft(spec)
    print type(rawData[0])
    print type(rawData[0])
    print type(reCon[0])
    wave.write("spring_re.wav", rate, reCon)
    print type(rawData), (reCon)
    print rawData.shape, reCon.shape
    '''
