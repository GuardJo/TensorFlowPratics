import tensorflow as tf

file_queue = tf.train.string_input_producer(['framingham.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(file_queue)
record_default = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_default)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=100)

x = tf.placeholder(tf.float32, shape=[None, 15])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.multiply(tf.Variable(tf.random_normal([15, 1]), name='weight'), [0.01])
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(1001):
        x_data, y_data = sess.run([train_x_batch, train_y_batch])
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})

    print("\nHypothesis : ", h, "\nCorrect(y) : ", c, "\nAccuracy : ", a)

    coord.request_stop()
    coord.join(threads)



