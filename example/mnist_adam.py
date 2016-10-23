"""Tinyflow example code.

This code is adapted from Tensorflow's MNIST Tutorial with minimum code changes.
"""
import tinyflow as tf
from tinyflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y = tf.nn.softmax(tf.matmul(x, W))

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

sess = tf.Session('gpu fusion')
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    res = sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(correct_prediction)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
