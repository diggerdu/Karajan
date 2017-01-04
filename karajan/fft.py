import tensorflow as tf
import numpy as np
import scipy


def stft(wav, n_fft=1024, overlap=4, dt=tf.int32, absp=False):
    assert (wav.shape[0] > n_fft)
    X = tf.placeholder(dtype=dt,shape=wav.shape)
    X = tf.cast(X,tf.float32)
    hop = n_fft / overlap
    
    ## prepare constant variable
    Pi = tf.constant(np.pi, dtype=tf.float32)
    W = tf.constant(scipy.hanning(n_fft), dtype=tf.float32)
    S = tf.pack([tf.fft(tf.cast(tf.multiply(W,X[i:i+n_fft]),tf.complex64)) for i in range(1, wav.shape[0] - n_fft, hop)])
    sess = tf.Session()
    hahaha = sess.run(S,feed_dict={X:wav})
    return hahaha
if __name__ == '__main__':
    a = np.arange(30000)
    print stft(a).shape


