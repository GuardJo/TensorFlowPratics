import tensorflow as tf
import numpy as np


with tf.Session() as sess:
    # one_hot example
    sess.run(tf.global_variables_initializer())
    t = tf.one_hot([[0], [1], [2], [0]], depth=3)
    print(sess.run(t))
    t = tf.reshape(t, shape=[-1, 3])
    print(sess.run(t))

    # Casting example
    arr = np.array([1.8, 2.2, 3.3, 4.9])
    t2 = tf.cast(arr, tf.int32)
    print(sess.run(t2))
    t3 = tf.cast([True, False, 1 == 1, 0 == 1], tf.int32)
    print(sess.run(t3))

    # Stack example
    x = [1, 4]
    y = [2, 5]
    z = [3, 6]
    s1 = tf.stack([x, y, z])
    s2 = tf.stack([x, y, z], axis=1)
    print(sess.run(s1))
    print(sess.run(s2))
    one = tf.ones_like(s1)
    zero = tf.zeros_like(s1)
    print(sess.run(one))
    print(sess.run(zero))

