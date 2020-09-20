import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
l1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
l3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')
l3 = tf.nn.relu(l3)
l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l3 = tf.nn.dropout(l3, keep_prob=keep_prob)
l3 = tf.reshape(l3, [-1, 128 * 4 * 4])

w4 = tf.get_variable("w4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
l4 = tf.matmul(l3, w4) + b4
l4 = tf.nn.relu(l4)
l4 = tf.nn.dropout(l4, keep_prob=keep_prob)

w5 = tf.get_variable("w5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l4, w5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Learning started, It takes sometime")
training_epochs = 15
batch_size = 100

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict= feed_dict)
        avg_cost += c / total_batch
    print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
print("Learning Finished")

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

