import tinyflow as tf
from tinyflow.datasets import get_cifar10
import resnet_model

batch_size = 100
num_classes = 10

hps = resnet_model.HParams(batch_size=batch_size,
                           num_classes=num_classes,
                           min_lrn_rate=0.0001,
                           lrn_rate=0.1,
                           num_residual_units=5,
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

known_shape = {model._images: [batch_size, 3, 32, 32], model.labels: [batch_size, num_classes]}
stdev = 0.1
init_step = []
for v, name, shape in tf.infer_variable_shapes(
    model.train_op, feed_dict=known_shape):
    init_step.append(tf.assign(v, tf.normal(shape, stdev)))
    print("shape[%s]=%s" % (name, shape))
sess.run(init_step)

cifar = get_cifar10()

num_batch = 600
for epoch in range(10):
    print("epoch[%d]" % epoch)
    sum_loss = 0.0
    for i in range(num_batch):
        batch_xs, batch_ys = cifar.train.next_batch(100)
        loss, _ = sess.run([model.cost, model.train_op],
                feed_dict={x: batch_xs, label: batch_ys})
        sum_loss += loss
    print("epoch[%d] loss=%g" % (epoch, sum_loss /num_batch))

