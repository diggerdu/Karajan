import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.int16, shape=[10])
x = tf.slice(a,[0],[4])
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
result = sess.run(x, feed_dict={a:np.arange(10)})

print result
