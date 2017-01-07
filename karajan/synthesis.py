import tensorflow as tf
import numpy as np


def spec2wav(amp, iters=1000):
    a = tf.placeholder(tf.float32, shape=amp.shape)
    Pi = tf.constant(np.pi, dtype=tf.float32)
    ang = 2 * Pi * tf.random_uniform(amp.shape, dtype=tf.float32)
    for i in range(iters):
        S = tf.matmul(a * tf.exp(1j * ang)) 
        W = tf
    
