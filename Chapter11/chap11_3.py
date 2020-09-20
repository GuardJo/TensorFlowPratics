# Max Pooling
import tensorflow as tf
import numpy as np

interactive_sess = tf.InteractiveSession()
image = np.array([[[[4], [3]],
                   [[2], [1]]]])
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

print(pool.shape)
print(pool.eval())