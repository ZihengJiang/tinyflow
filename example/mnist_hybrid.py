"""Tinyflow example code.

This code is adapted from Tensorflow's MNIST Tutorial with minimum code changes.
"""
import tinyflow as tf
from tinyflow.datasets import get_mnist

def my_sigmoid(x):
    return 1 / (1 + tf.exp(-x))

# Create the model
x  = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.normal([784, 512], 0.01))
W2 = tf.Variable(tf.normal([512, 10],  0.01))

m  = tf.matmul(x, W1)
h1 = my_sigmoid(tf.matmul(x, W1))
y  = tf.nn.softmax(tf.matmul(h1, W2))

# Define loss and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session("gpu fusion")
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(correct_prediction)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
