import tensorflow as tf
import numpy as np
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope("layer1") as scope:
    w1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer = tf.sigmoid(tf.matmul(x, w1) + b1)
    w1_hist = tf.summary.histogram("weight1", w1)
    b1_hist = tf.summary.histogram("weight2", b1)
    layer_hist = tf.summary.histogram("layer", layer)

with tf.name_scope("layer2") as scope:
    w2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer, w2) + b2)
    w2_hist = tf.summary.histogram("weight2", w2)
    b2_hist = tf.summary.histogram("bias2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

cost_summ = tf.summary.scalar("cost", cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

acc_summ = tf.summary.scalar("accuracy", accuracy)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs")
    writer.add_graph(sess.graph)
    for step in range(10001):
        s, cost_val, _ = sess.run([summary, cost, train], feed_dict={x: x_data, y: y_data})
        writer.add_summary(s, global_step=step)

        if step % 100 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, prediction, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis : ", h, "\nCorrect : ", c, "\nAccuracy : ", a)
