import tensorflow as tf

file_name_queue = tf.train.string_input_producer(["data_sample.csv"], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(file_name_queue)
record_default = [[0], [0], [0], [0], [0], [0], [0]]
xy = tf.decode_csv(value, record_defaults=record_default)
train_x_batch, train_y_batch = tf.train.batch([xy[0:-3], xy[-3:]], batch_size=10)

x = tf.placeholder("float", shape=[None, 4])
y = tf.placeholder("float", shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(2001):
        x_data, y_data = sess.run([train_x_batch, train_y_batch])
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}))

    all = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))

    coord.request_stop()
    coord.join(threads)