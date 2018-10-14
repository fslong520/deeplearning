import tensorflow as tf
a = tf.placeholder(tf.float32, name='input_1')
b = tf.placeholder(tf.float32, name='input_2')
output = tf.multiply(a, b, name='output')
input_dict = {a: 7.0, b: 10.0}
with tf.Session() as sess:
    print(sess.run(output, feed_dict=input_dict))
