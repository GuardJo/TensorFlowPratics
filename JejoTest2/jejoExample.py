import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

arr = np.loadtxt("test3.csv", delimiter=',', dtype=np.object, encoding='utf-8')
arr = MinMaxScaler().fit_transform(arr)
arr2 = np.loadtxt("test3_2.csv", delimiter=',', dtype=np.object, encoding='utf-8')
arr2 = MinMaxScaler().fit_transform(arr2)

x_data = arr[0:, 1: -1]

x_test = arr2[0:, 1:-1]

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

    for epoch in range(1000) :
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})

        if (epoch + 1) % 50 == 0:
            c = sess.run(cost, feed_dict={x: x_data, y: y_data})
            print(("Epoch : ", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(c), "w = ", sess.run(w), "b = ",
                   sess.run(b)))

    print("Complete")