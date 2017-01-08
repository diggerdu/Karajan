import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda.fft import *
import scipy, pylab
import numpy as np
import scipy.io.wavfile as wave

def stft(x, fftsize=1024, overlap=4):   
    x = x[:1600*10]
    hop = fftsize / overlap
    w = scipy.hanning(fftsize)  
    unroll_x = np.asarray([w*x[i:i+fftsize] for i in range(0, len(x)-fftsize, hop)], np.float32)
    plan = Plan(shape=fftsize, in_dtype=np.float32, out_dtype=np.complex64, batch=unroll_x.shape[0], idist=fftsize, odist=fftsize/2+1)
    print unroll_x.shape
    x_gpu = gpuarray.to_gpu(unroll_x.flatten())
    xf_gpu = gpuarray.empty(fftsize/2+1*unroll_x.shape[1], np.complex64)
    fft(x_gpu, xf_gpu, plan)
    s = x_gpu.get()
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
    spec = stft(rawData, fftsize=256)
    
    '''    reCon = istft(spec)
    print type(rawData[0])
    print type(rawData[0])
    print type(reCon[0])
    wave.write("spring_re.wav", rate, reCon)
    print type(rawData), (reCon)
    print rawData.shape, reCon.shape
    '''
