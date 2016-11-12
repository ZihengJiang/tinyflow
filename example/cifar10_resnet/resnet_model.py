# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from nnvm import graph
import tinyflow as tf
import numpy as np
from collections import namedtuple


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(tf.zeros(shape=[1]), name='global_step')
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual
    filters = [16, 16, 32, 64]

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])

    for i in xrange(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])

    for i in xrange(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])

    for i in xrange(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x, filters[3])
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      self.cost = tf.nn.mean_sparse_softmax_cross_entropy_with_logits(
          logits, self.labels)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = self.hps.lrn_rate

    optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    apply_op = optimizer.minimize(self.cost)
    train_ops = [apply_op]
    self.train_op = tf.group(*train_ops)

  def _batch_norm(self, name, x, num_filter):
    """Batch normalization."""
    with tf.variable_scope(name):
      gamma = tf.get_variable("bn_gamma", tf.normal([num_filter], 1.0))
      beta  = tf.get_variable("bn_beta",  tf.zeros([num_filter]))
      y = tf.nn.batch_normalization(x, gamma, beta)
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x, in_filter)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x, in_filter)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x, out_filter)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, ksize=stride, strides=stride, padding='VALID')
        orig_x = tf.pad(orig_x, dim=1, pad=(out_filter-in_filter)//2)
        orig_x = tf.pad(orig_x, dim=1, pad=-(out_filter-in_filter)//2)
      x += orig_x

    return x

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('conv_DW',
                tf.normal([out_filters, in_filters, filter_size, filter_size], np.sqrt(2/n)))
        return tf.nn.conv2d(x, kernel, num_filter=out_filters, strides=strides)

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.nn.flatten_layer(x)
    w = tf.get_variable('fc_DW')
    b = tf.get_variable('biases', tf.zeros([out_dim]))
    return tf.nn.linear(x, w, b, num_hidden=out_dim, no_bias=False)

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.leaky_relu(x, leakiness=leakiness)

  def _global_avg_pool(self, x):
    return tf.reduce_mean(x, reduction_indices=[1, 2])
