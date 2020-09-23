import tensorflow as tf
import numpy as np

arr = np.loadtxt('output.txt', delimiter=',', dtype=np.object)
x_data = arr[1:, 0:-1]
y_data = arr[1:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 5])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict={x: x_data, y: y_data})

        if step % 100 == 0:
            print(step, "Cost : ", cost_val)