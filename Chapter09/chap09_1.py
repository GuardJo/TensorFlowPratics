import tensorflow as tf
import numpy as np
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
w1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer = tf.sigmoid(tf.matmul(x, w1) + b1)
w2 = tf.Variable(tf.random_normal([10, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')

hypothesis = tf.sigmoid(tf.matmul(layer, w2) + b2)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run([w1, w2]))

    h, c, a = sess.run([hypothesis, prediction, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis : ", h, "\nCorrect : ", c, "\nAccuracy : ", a)

