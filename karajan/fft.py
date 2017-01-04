import tensorflow as tf
import numpy as np
import scipy


def stft(wav, n_fft=1024, overlap=4, dt=tf.int16, absp=False):
    assert (wav.shape[0] > n_fft)
    X = tf.plcaeholder(dtype=dt,shape=wav.shape)
    S = tf.fft(X)
    hop = n_fft / overlap
    
    ## prepare constant variable
    Pi = tf.constant(np.pi, dtype=tf.float32)
    W = tf.constant(scipy.hanning(n_fft), dtype=tf.float32)
    
