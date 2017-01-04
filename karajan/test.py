import tensorflow as tf


a = tf.placeholder(tf.int16, shape=[10])
x = tf.strided_slice(a, [0,], [9,])
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
result = sess.run(x, feed_dict={a:[1]})

print result
