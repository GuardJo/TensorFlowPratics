import tensorflow as tf
# axis example


x = [[0, 1, 2], [2, 1, 0]]


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(tf.argmax(x, axis=1)))