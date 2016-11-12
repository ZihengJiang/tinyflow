import tinyflow as tf
from tinyflow.datasets import get_cifar10
import resnet_model

batch_size = 32
num_epoch = 100
num_batch = 600
num_classes = 10

hps = resnet_model.HParams(batch_size=batch_size,
                           num_classes=num_classes,
                           min_lrn_rate=0.0001,
                           lrn_rate=0.1,
                           num_residual_units=10,
                           use_bottleneck=False,
                           weight_decay_rate=0.0002,
                           relu_leakiness=0.1,
                           optimizer='sgd')

x     = tf.placeholder(tf.float32)
label = tf.placeholder(tf.float32)
model = resnet_model.ResNet(hps, x, label, "train")
model.build_graph()


sess = tf.Session("gpu")
init = tf.initialize_all_variables()
sess.run(init)

known_shape = {model._images: [batch_size, 3, 32, 32], model.labels: [batch_size]}
stdev = 0.1
init_step = []
for v, name, shape in tf.infer_variable_shapes(
    model.train_op, feed_dict=known_shape):
    init_step.append(tf.assign(v, tf.normal(shape, stdev)))
    print("shape[%s]=%s" % (name, shape))
sess.run(init_step)

cifar = get_cifar10()
g_batch_xs, g_batch_ys = cifar.train.next_batch(batch_size)

for epoch in range(num_epoch):
    print("epoch[%d]" % epoch)
    sum_loss = 0.0
    for i in range(num_batch):
        batch_xs, batch_ys = cifar.train.next_batch(batch_size)
        loss, _ = sess.run([model.cost, model.train_op],
                feed_dict={x: batch_xs, label: batch_ys})
        sum_loss += loss
    print("epoch[%d] loss=%g" % (epoch, sum_loss /num_batch))

correct_prediction = tf.equal(tf.argmax(model.predictions, 1), label)
accuracy = tf.reduce_mean(correct_prediction)
print(sess.run(accuracy, feed_dict={x: cifar.test.images, label: cifar.test.labels}))


def train_epoch(num_epoch):
    for epoch in range(num_epoch):
        print("epoch[%d]" % epoch)
        sum_loss = 0.0
        for i in range(num_batch):
            batch_xs, batch_ys = cifar.train.next_batch(batch_size)
            loss, _ = sess.run([model.cost, model.train_op],
                    feed_dict={x: g_batch_xs, label: g_batch_ys})
            sum_loss += loss
        print("epoch[%d] loss=%g" % (epoch, sum_loss /num_batch))

def train():
    for i in range(n):
        # batch_xs, batch_ys = cifar.train.next_batch(batch_size)
        pred, loss, _ = sess.run([model.predictions, model.cost, model.train_op],
                feed_dict={x: g_batch_xs, label: g_batch_ys})
    return pred, loss

def evaluate():
    correct_prediction = tf.equal(tf.argmax(model.predictions, 1), label)
    accuracy = tf.reduce_mean(correct_prediction)
    print(sess.run(accuracy, feed_dict={x: g_batch_xs, label: g_batch_ys}))
