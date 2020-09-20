import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
training_epochs = 15
batch_size = 100

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, [None, 784])
            x_img = tf.reshape(self.x, [-1, 28, 28, 1])
            self.y = tf.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)

            l1 = tf.layers.conv2d(inputs=x_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            l1 = tf.layers.max_pooling2d(inputs=l1, pool_size=[2, 2], padding='SAME', strides=2)
            l1 = tf.layers.dropout(inputs=l1, rate=0.3, training=self.training)

            l2 = tf.layers.conv2d(inputs=l1, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            l2 = tf.layers.max_pooling2d(inputs=l2, pool_size=[2, 2], padding='SAME', strides=2)
            l2 = tf.layers.dropout(inputs=l2, rate=0.3, training=self.training)

            l3 = tf.layers.conv2d(inputs=l2, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            l3 = tf.layers.max_pooling2d(inputs=l3, pool_size=[2, 2], padding='SAME', strides=2)
            l3 = tf.layers.dropout(inputs=l3, rate=0.3, training=self.training)
            l3 = tf.reshape(l3, [-1, 4 * 4 * 128])

            l4 = tf.layers.dense(inputs=l3, units=625, activation=tf.nn.relu)
            l4 = tf.layers.dropout(inputs=l4, rate=0.3, training=self.training)

            self.logits = tf.layers.dense(inputs=l4, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.x: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x: x_data, self.y: y_data, self.training: training})

sess = tf.Session()
models = []
num_models = 7
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print("Learning Started!")

for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
    print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', avg_cost_list)

print("Learning Finished!")

test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy : ', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy : ', sess.run(ensemble_accuracy))


