import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

arr = np.loadtxt("test3.csv", delimiter=',', dtype=np.object, encoding='utf-8')

x_data = arr[1:, 1: -1]
x_data = MinMaxScaler().fit_transform(x_data)
y_data = arr[1:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cost)

prediction = tf.equal(hypothesis, y)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost, optimizer], feed_dict={x: x_data, y: y_data})

        if step % 100 == 0:
            print("step : ", step, "cost : ", cost_val)

    # p, a = sess.run([prediction, accuracy], feed_dict={x: x_data, y: y_data})
    # print("Prediction : ", p, "Accuracy : ", a)

    hy_val, y_val = sess.run([hypothesis, y], feed_dict={x: x_data, y: y_data})
    print("hypothesis : ", hy_val, "y : ", y_val)
