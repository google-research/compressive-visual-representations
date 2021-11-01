# coding=utf-8
# Copyright 2020 The Compressive Visual Representations Authors.
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
"""Network architectures related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import flags
from compressive_visual_representations import resnet
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5


def add_weight_decay(model, adjust_per_optimizer=True):
  """Compute weight decay from flags."""
  if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
    # Weight decay are taking care of by optimizer for these cases.
    # Except for supervised head, which will be added here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_variables
        if 'head_supervised' in v.name and 'bias' not in v.name
    ]
    if l2_losses:
      return FLAGS.weight_decay * tf.add_n(l2_losses)
    else:
      return 0

  # TODO(srbs): Think of a way to avoid name-based filtering here.
  l2_losses = [
      tf.nn.l2_loss(v)
      for v in model.trainable_weights
      if 'batch_normalization' not in v.name
  ]
  loss = FLAGS.weight_decay * tf.add_n(l2_losses)
  return loss


def get_train_steps(num_examples):
  """Determine the number of training steps."""
  return FLAGS.train_steps or (
      num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)


def cosine_ramping_schedule(base_value, num_examples):
  """Cosine ramping schedule."""
  total_steps = get_train_steps(num_examples)
  cosine_decay = tf.keras.experimental.CosineDecay(1. - base_value, total_steps)
  def schedule_fn(global_step):
    value = 1. - cosine_decay(global_step)
    return value
  return schedule_fn


def cosine_ramping_from_zero_schedule(target_value, num_examples):
  """Cosine ramping schedule."""
  total_steps = get_train_steps(num_examples)
  cosine_decay = tf.keras.experimental.CosineDecay(target_value, total_steps)
  def schedule_fn(global_step):
    value = target_value - cosine_decay(global_step)
    return value
  return schedule_fn


def warmup_schedule(target_value, warmup_steps=None):
  """Linear ramping schedule."""
  def schedule_fn(global_step):
    if warmup_steps is not None and warmup_steps > 0:
      value = global_step / int(warmup_steps) * target_value
      value = tf.where(global_step < warmup_steps, value, target_value)
    else:
      value = target_value
    return value
  return schedule_fn


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self, base_learning_rate, num_examples, name=None):
    super(WarmUpAndCosineDecay, self).__init__()
    self.base_learning_rate = base_learning_rate
    self.num_examples = num_examples
    self._name = name

  def __call__(self, step):
    with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
      warmup_steps = int(
          round(FLAGS.warmup_epochs * self.num_examples //
                FLAGS.train_batch_size))
      if FLAGS.learning_rate_scaling == 'linear':
        scaled_lr = self.base_learning_rate * FLAGS.train_batch_size / 256.
      elif FLAGS.learning_rate_scaling == 'sqrt':
        scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_batch_size)
      else:
        raise ValueError('Unknown learning rate scaling {}'.format(
            FLAGS.learning_rate_scaling))
      learning_rate = (
          step / int(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

      # Cosine decay learning rate schedule
      total_steps = get_train_steps(self.num_examples)
      # TODO(srbs): Cache this object.
      cosine_decay = tf.keras.experimental.CosineDecay(
          scaled_lr, total_steps - warmup_steps)
      learning_rate = tf.where(step < warmup_steps, learning_rate,
                               cosine_decay(step - warmup_steps))

      return learning_rate

  def get_config(self):
    return {
        'base_learning_rate': self.base_learning_rate,
        'num_examples': self.num_examples,
    }


class LinearLayer(tf.keras.layers.Layer):
  """Linear layer."""

  def __init__(
      self,
      num_classes,
      use_bias=True,
      use_bn=False,
      kernel_initializer=None,
      name='linear_layer',
      **kwargs):
    super(LinearLayer, self).__init__(**kwargs)
    if kernel_initializer is None:
      if FLAGS.mlp_init == 'simclr':
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
      elif FLAGS.mlp_init == 'byol':
        kernel_initializer = tf.keras.initializers.VarianceScaling()
      else:
        raise NotImplementedError

    self.num_classes = num_classes
    self.use_bn = use_bn
    self._name = name
    if callable(self.num_classes):
      num_classes = -1
    else:
      num_classes = self.num_classes
    self.dense = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=kernel_initializer,
        # kernel_initializer=tf.compat.v1.variance_scaling_initializer(1e-4),
        use_bias=use_bias and not self.use_bn)
    if self.use_bn:
      self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

  def build(self, input_shape):
    # TODO(srbs): Add a new SquareDense layer.
    if callable(self.num_classes):
      self.dense.units = self.num_classes(input_shape)
    super(LinearLayer, self).build(input_shape)

  def call(self, inputs, training):
    assert inputs.shape.ndims == 2, inputs.shape
    inputs = self.dense(inputs)
    if self.use_bn:
      inputs = self.bn_relu(inputs, training=training)
    return inputs


class ProjectionHead(tf.keras.layers.Layer):
  """Projection head."""

  def __init__(self, trainable=True, num_outputs_per_layer=None,
               out_dim=128, use_bn_mid=True, name_prefix='', **kwargs):
    # out_dim = FLAGS.proj_out_dim
    self.linear_layers = []
    if FLAGS.proj_head_mode == 'none':
      pass  # directly use the output hiddens as hiddens
    elif FLAGS.proj_head_mode == 'linear':
      self.linear_layers = [
          LinearLayer(
              num_classes=out_dim, use_bias=False,
              use_bn=FLAGS.proj_head_output_bn, name=name_prefix + 'l_0',
              trainable=trainable)
      ]
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          if num_outputs_per_layer is not None:
            num_classes = num_outputs_per_layer[j]
          else:
            # Just use the same output size as the input.
            num_classes = lambda input_shape: int(input_shape[-1])
          self.linear_layers.append(
              LinearLayer(
                  num_classes=num_classes,
                  use_bias=True,
                  use_bn=use_bn_mid,
                  trainable=trainable,
                  name=name_prefix + 'nl_%d' % j))
        else:
          # for the final layer, neither bias nor relu is used.
          self.linear_layers.append(
              LinearLayer(
                  num_classes=out_dim,
                  use_bias=False,
                  use_bn=FLAGS.proj_head_output_bn,
                  trainable=trainable,
                  name=name_prefix + 'nl_%d' % j))
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    super(ProjectionHead, self).__init__(**kwargs)

  def call(self, inputs, training):
    if FLAGS.proj_head_mode == 'none':
      return inputs, inputs  # directly use the output hiddens as hiddens
    hiddens_list = [tf.identity(inputs, 'proj_head_input')]
    del inputs  # Use hiddens_list from here on.
    if FLAGS.proj_head_mode == 'linear':
      assert len(self.linear_layers) == 1, len(self.linear_layers)
      return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                       training))
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        hiddens = self.linear_layers[j](hiddens_list[-1], training)
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          hiddens = tf.nn.relu(hiddens)
        hiddens_list.append(hiddens)
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    # The first element is the output of the projection head.
    # The second element is the input of the finetune head.
    proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
    supervised_head_input = tf.identity(hiddens_list[FLAGS.ft_proj_selector],
                                        'supervised_head_input')
    return proj_head_output, supervised_head_input


class StochasticProjectionHead(tf.keras.layers.Layer):
  """Projection head with an extra stochastic layer."""

  def __init__(self, trainable=True, num_outputs_per_layer=None,
               out_dim=256, use_bn_mid=True, extra_linear_layer=True,
               name_prefix='', **kwargs):
    self.extra_linear_layer = extra_linear_layer
    self.linear_layers = []
    for j in range(FLAGS.num_proj_layers):
      if j != FLAGS.num_proj_layers - 1:
        # for the middle layers, use bias and relu for the output.
        if num_outputs_per_layer is not None:
          num_classes = num_outputs_per_layer[j]
        else:
          # Just use the same output size as the input.
          num_classes = lambda input_shape: int(input_shape[-1])
        self.linear_layers.append(
            LinearLayer(
                num_classes=num_classes,
                use_bias=True,
                use_bn=use_bn_mid,
                trainable=trainable,
                name=name_prefix + 'nl_%d' % j))
      else:
        if self.extra_linear_layer:
          self.linear_layers.append(
              LinearLayer(
                  num_classes=out_dim,
                  use_bias=False,
                  use_bn=False,
                  trainable=trainable,
                  name=name_prefix + 'nl_%d' % j))
        else:
          self.linear_layers.append(
              LinearLayer(
                  num_classes=out_dim,
                  use_bias=False,
                  use_bn=False,
                  trainable=trainable,
                  name=name_prefix + 'nl_%d' % j))
      if self.extra_linear_layer:
        self.linear_layers.append(
            LinearLayer(
                num_classes=out_dim,
                use_bias=False,
                use_bn=False,
                trainable=trainable,
                name=name_prefix + 'nl_%d' % FLAGS.num_proj_layers))
    super(StochasticProjectionHead, self).__init__(**kwargs)

  def call(self, inputs, training):
    hiddens_list = [tf.identity(inputs, 'proj_head_input')]
    del inputs  # Use hiddens_list from here on.
    for j in range(FLAGS.num_proj_layers-1):
      hiddens = self.linear_layers[j](hiddens_list[-1], training)
      hiddens = tf.nn.relu(hiddens)
      hiddens_list.append(hiddens)
    if self.extra_linear_layer:
      hiddens = self.linear_layers[-2](hiddens_list[-1], training)
      hiddens_list.append(hiddens)
    mu = tf.math.l2_normalize(hiddens, -1)
    if training:
      z_dist = tfd.VonMisesFisher(mu, FLAGS.kappa_e)
      z = z_dist.sample()
    else:
      z = mu
    hiddens = z
    z_mu = tf.concat([z, mu], axis=-1)
    hiddens = self.linear_layers[-1](hiddens, training)
    hiddens_list.append(hiddens)
    # The first element is the output of the projection head.
    # The second element is the input of the finetune head.
    proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
    return proj_head_output, z_mu


class SupervisedHeadForFinetuning(tf.keras.layers.Layer):
  """Supervised head for finetuning."""

  def __init__(self, num_classes, name='head_supervised_ft', **kwargs):
    super(SupervisedHeadForFinetuning, self).__init__(name=name, **kwargs)
    if FLAGS.zero_init_logits_layer:
      self.linear_layer = LinearLayer(
          num_classes, kernel_initializer=tf.keras.initializers.Zeros())
    else:
      self.linear_layer = LinearLayer(num_classes)

  def call(self, inputs, training):
    inputs = self.linear_layer(inputs, training)
    inputs = tf.identity(inputs, name='logits_sup')
    return inputs


class SupervisedHead(tf.keras.layers.Layer):
  """Supervised head."""

  def __init__(self, num_classes, name='head_supervised', **kwargs):
    super(SupervisedHead, self).__init__(name=name, **kwargs)
    if FLAGS.zero_init_logits_layer:
      self.linear_layer = LinearLayer(
          num_classes, kernel_initializer=tf.keras.initializers.Zeros())
    else:
      self.linear_layer = LinearLayer(num_classes)
    # for var in tf.trainable_variables():
    #   if var.name.startswith(name):
    #     tf.add_to_collection('trainable_variables_inblock_5', var)

  def call(self, inputs, training):
    inputs = self.linear_layer(inputs, training)
    inputs = tf.identity(inputs, name='logits_sup')
    return inputs


class BatchNorm(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               relu=True,
               init_zero=False,
               center=True,
               scale=True,
               data_format='channels_last',
               **kwargs):
    super(BatchNorm, self).__init__(**kwargs)
    self.relu = relu
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()
    if data_format == 'channels_first':
      axis = 1
    else:
      axis = -1
    if FLAGS.global_bn:
      # TODO(srbs): Set fused=True
      # Batch normalization layers with fused=True only support 4D input
      # tensors.
      self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
          axis=axis,
          momentum=FLAGS.batch_norm_decay,
          epsilon=BATCH_NORM_EPSILON,
          center=center,
          scale=scale,
          gamma_initializer=gamma_initializer)
    else:
      # TODO(srbs): Set fused=True
      # Batch normalization layers with fused=True only support 4D input
      # tensors.
      self.bn = tf.keras.layers.BatchNormalization(
          axis=axis,
          momentum=FLAGS.batch_norm_decay,
          epsilon=BATCH_NORM_EPSILON,
          center=center,
          scale=scale,
          fused=False,
          gamma_initializer=gamma_initializer)

  def call(self, inputs, training):
    inputs = self.bn(inputs, training=training)
    return inputs
