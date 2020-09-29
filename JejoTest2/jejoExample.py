import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

arr = np.loadtxt("test2_2.csv", delimiter=',', dtype=np.object, encoding='utf-8')
xy = MinMaxScaler().fit_transform(arr)
x_data = xy[1:, 1: -1]
y_data = xy[1:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 10]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.equal(hypothesis, y)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, hy_val, y_val, _ = sess.run([cost, hypothesis, y, optimizer], feed_dict={x: x_data, y: y_data})

        if step % 500 == 0:
            print("step : ", step, "cost : ", cost_val)

    # p, c = sess.run([prediction, accuracy], feed_dict={x: x_data, y: y_data})
    # print("prediction Test", p, "accuracy : ", c)

    hy_val, y_val = sess.run([hypothesis, y], feed_dict={x: x_data, y: y_data})

    print("hypothesis : ", hy_val, "y_value : ", y_val)