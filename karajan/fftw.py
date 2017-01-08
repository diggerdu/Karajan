import tensorflow as tf
import numpy as np
import scipy
import time

def stft(wav, n_fft=1024, overlap=4, dt=tf.int32, absp=False):
    assert (wav.shape[0] > n_fft)
    X = tf.placeholder(dtype=dt,shape=wav.shape)
    X = tf.cast(X,tf.float32)
    hop = n_fft / overlap
    
    ## prepare constant variable
    Pi = tf.constant(np.pi, dtype=tf.float32)
    W = tf.constant(scipy.hanning(n_fft), dtype=tf.float32)
    start = time.time()
    X_W = [tf.cast(tf.multiply(W,X[i:i+n_fft]),\
            tf.complex64) for i in xrange(1, wav.shape[0] - n_fft, hop)]
    print 'tensorflow building the graph:',time.time() - start
    '''
    abs_S = tf.complex_abs(S)
    sess = tf.Session()
    start = time.time()
    '''
    if absp:
        return sess.run(abs_S, feed_dict={X:wav})
    else:
        return sess.run(X_W, feed_dict={X:wav})
    print 'the model actually run:',time.time()-start

def istft(spec, overlap=4):
    assert (spec.shape[0] > 1)
    S = placeholder(dtype=tf.complex64, shape=spec.shape)
    X = tf.complex_abs(tf.concat(0, [tf.ifft(frame) \
            for frame in tf.unstack(S)]))
    sess = tf.Session()
    return sess.run(X, feed_dict={S:spec})
if __name__ == '__main__':
    a = np.ones((44100*600))
    s = time.time()
    print stft(a).shape
    print 'tensorflow stft:',time.time() - s
    s = time.time()
    import librosa
    print librosa.stft(a).shape
    print 'librosa stft', time.time() -s

