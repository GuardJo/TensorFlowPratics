import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
l1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.reshape(l2, [-1, 7 * 7 * 64])

w3 = tf.get_variable("w3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l2, w3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

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
        feed_dict = {x: batch_xs, y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict= feed_dict)
        avg_cost += c / total_batch
    print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
print("Learning Finished")

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

