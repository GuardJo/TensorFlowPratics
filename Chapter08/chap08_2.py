import tensorflow as tf
import numpy as np

# reshape example

t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])


t2 = tf.reshape(t, shape=[-1, 3])

t3 = tf.reshape(t, shape=[-1, 2, 3])

t4 = tf.squeeze([0, 1, 2])

t5 = tf.expand_dims([0, 1, 2], 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(t.__str__())
print(sess.run(t2))
print(sess.run(t3))
print(sess.run(t4))
print(sess.run(t5))

