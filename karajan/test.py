import tensorflow as tf


a = tf.placeholder(tf.int16, shape=[1])
x = a
for i in range(6):
    x = tf.add(x,x)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
result = sess.run(x, feed_dict={a:[1]})

print result
