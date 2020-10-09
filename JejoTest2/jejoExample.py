import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

arr = np.loadtxt("test3_1.csv", delimiter=',', dtype=np.object, encoding='utf-8')
arr2 = np.loadtxt("test3_2.csv", delimiter=',', dtype=np.object, encoding='utf-8')

x_data = arr[0:, 1: -1]
x_data = MinMaxScaler().fit_transform(x_data)

x_test = arr2[0:, 1:-1]
x_test = MinMaxScaler().fit_transform(x_test)

y_data = arr[0:, [-1]]

y_test = arr2[0:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)

prediction = tf.equal(hypothesis, y)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(50001):
        cost_val, _ = sess.run([cost, optimizer], feed_dict={x: x_data, y: y_data})

        if step % 1000 == 0:
            print("step : ", step, "cost : ", cost_val)

    # p, a = sess.run([prediction, accuracy], feed_dict={x: x_data, y: y_data})
    # print("Prediction : ", p, "Accuracy : ", a)

    hy_val, w_val, b_val = sess.run([hypothesis, w, b], feed_dict={x: x_test, y: y_test})
    print("hypothesis : \n", hy_val, "w_val : \n", w_val, "b_val : \n", b_val)
