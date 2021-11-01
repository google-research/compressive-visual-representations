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
"""Model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from absl import flags
from absl import logging

from compressive_visual_representations import data_util
from compressive_visual_representations import lars_optimizer
from compressive_visual_representations import model_util
from compressive_visual_representations import resnet

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

FLAGS = flags.FLAGS


def natural_key(string):
  """Key to use for "natural sorting".

  Based on https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/

  Args:
    string: The input string to produce key for

  Returns:
    key to use for natural sorting in conjunction with Python's built-in sort.
  """
  return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', string)]


def arg_natural_sort(array):
  return [idx for idx, val in sorted(
      enumerate(array), key=lambda x: natural_key(x[1]))]


class Model(tf.keras.models.Model):
  """Resnet model with projection or supervised layer."""
  # TODO(leekh): maybe support kappa heads in the momentum encoder setting

  def __init__(self, resnet_model, num_classes, target_encoder=None,
               use_momentum_proj_head=False, use_projector=False,
               use_target_projector=False, is_export_single_branch=False,
               **kwargs):
    super(Model, self).__init__(**kwargs)
    self._num_classes = num_classes
    self._target_resnet = target_encoder
    self._resnet = resnet_model
    self._use_momentum_proj_head = use_momentum_proj_head
    self._use_projector = use_projector
    self._use_target_projector = use_target_projector
    self.is_export_single_branch = is_export_single_branch

    if FLAGS.proj_head_hidden_dim is not None:
      num_hidden = FLAGS.num_proj_layers - 1
      num_outputs_per_layer = [FLAGS.proj_head_hidden_dim] * num_hidden
    else:
      num_outputs_per_layer = None

    self._projection_head = model_util.ProjectionHead(
        num_outputs_per_layer=num_outputs_per_layer,
        out_dim=FLAGS.proj_out_dim,
        name_prefix='head/')

    if self._use_projector:
      if FLAGS.stochastic_projector:
        self._projector = model_util.StochasticProjectionHead(
            num_outputs_per_layer=num_outputs_per_layer,
            out_dim=FLAGS.proj_out_dim,
            use_bn_mid=FLAGS.projector_mid_bn,
            name_prefix='projector/')
      else:
        self._projector = model_util.ProjectionHead(
            num_outputs_per_layer=num_outputs_per_layer,
            out_dim=FLAGS.proj_out_dim,
            use_bn_mid=FLAGS.projector_mid_bn,
            name_prefix='projector/')

    if self._target_resnet is not None:
      logging.info('Using target resnet encoder')
      self._weight_permutation = []
      self._target_head = model_util.ProjectionHead(
          trainable=True,
          num_outputs_per_layer=num_outputs_per_layer,
          out_dim=FLAGS.proj_out_dim,
          name_prefix='target_head_ema/'
          if use_momentum_proj_head else 'target_head/')
      if use_momentum_proj_head:
        self._head_weight_permutation = []
      if self._use_target_projector:
        self._target_projector = model_util.ProjectionHead(
            num_outputs_per_layer=num_outputs_per_layer,
            out_dim=FLAGS.proj_out_dim,  # FLAGS.proj_out_dim
            use_bn_mid=FLAGS.projector_mid_bn,
            name_prefix='target_projector/')

    if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
      if FLAGS.train_mode == 'finetune' and FLAGS.rename_supervised_head:
        self.actual_supervised_head = model_util.SupervisedHead(
            self._num_classes, name='ft_head_supervised')
        # After the checkpoint is used, we are going to reassign
        # self.supervised_head = self.actual_supervised_head
        self.supervised_head = model_util.SupervisedHead(
            num_classes=1000, name='head_supervised')
        return
      self.supervised_head = model_util.SupervisedHead(self._num_classes)

  def reset_heads(self):
    if FLAGS.train_mode == 'finetune' and FLAGS.rename_supervised_head:
      logging.info('Reseting heads')
      self.supervised_head = self.actual_supervised_head

  def initialise_networks(self, input_shape):
    logging.info('Explicit weight initialization.')

    # online path
    inputs = tf.zeros(input_shape, dtype=tf.float32)
    hidden = self._resnet(inputs, training=False)
    hidden, supervised_input = self._projection_head(hidden, training=False)
    if self._use_projector:
      _, _ = self._projector(hidden, training=False)
    self.supervised_head(supervised_input, training=False)

    # target path
    if self._target_resnet is not None:
      hidden_target = self._target_resnet(inputs, training=False)
      hidden_target, _ = self._target_head(hidden_target, training=False)
      if self._use_target_projector:
        _, _ = self._target_projector(
            tf.math.l2_normalize(hidden_target, -1), training=False)

  def initialise_ema(self, input_shape):
    """Initialises weights of the EMA model."""

    logging.info('Explicit weight initialization.')

    # online path
    inputs = tf.zeros(input_shape, dtype=tf.float32)
    hidden = self._resnet(inputs, training=False)
    hidden, supervised_input = self._projection_head(hidden, training=False)
    if self._use_projector:
      _, _ = self._projector(hidden, training=False)
    self.supervised_head(supervised_input, training=False)

    # target path
    hidden_target = self._target_resnet(inputs, training=False)
    hidden_target, _ = self._target_head(hidden_target, training=False)
    if self._use_target_projector:
      _, _ = self._target_projector(
          tf.math.l2_normalize(hidden_target, -1),
          training=False)

    if not self._weight_permutation:
      # Get the permutations of the weights
      logging.info('Get the permutations of the resnet weights.')
      resnet_names = [x.name for x in self._resnet.trainable_weights]
      target_resnet_names = [
          x.name for x in self._target_resnet.trainable_weights
      ]

      resnet_indices = arg_natural_sort(resnet_names)
      target_resnet_indices = arg_natural_sort(target_resnet_names)

      assert len(target_resnet_indices) == len(
          resnet_indices), 'Model weights dont match'
      self._weight_permutation = [0] * len(target_resnet_indices)

      for t_idx, o_idx in zip(target_resnet_indices, resnet_indices):
        self._weight_permutation[t_idx] = o_idx

    if self._use_momentum_proj_head and not self._head_weight_permutation:
      # Get the permutations of the weights
      logging.info('Get the permutations of the head weights.')
      projection_head_names = [
          x.name for x in self._projection_head.trainable_weights
      ]
      target_head_names = [x.name for x in self._target_head.trainable_weights]

      projection_head_indices = arg_natural_sort(projection_head_names)
      target_head_indices = arg_natural_sort(target_head_names)
      assert len(target_head_indices) == len(
          projection_head_indices), 'Head weights dont match'
      self._head_weight_permutation = [0] * len(target_head_indices)

      for t_idx, o_idx in zip(target_head_indices, projection_head_indices):
        self._head_weight_permutation[t_idx] = o_idx

  def is_ema_initialised(self):
    """Returns True if EMA variables are initalised, or if EMA disabled."""
    if self._target_resnet is None:
      return True
    return (bool(self._target_resnet.trainable_weights) and
            bool(self._weight_permutation))

  def update_ema(self, ema_alpha):
    """Update the EMA model."""
    def _apply_moving(v_moving, v_normal):
      v_moving.assign_sub(
          tf.cast(1.- ema_alpha, v_moving.dtype) * (v_moving-v_normal))
      return v_moving

    def _update(strategy, v_moving_and_v_normal):
      for v_moving, v_normal in v_moving_and_v_normal:
        strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))

    ctx = tf.distribute.get_replica_context()
    resnet_weights = [
        self._resnet.trainable_weights[idx] for idx in self._weight_permutation]
    ctx.merge_call(_update, args=(
        zip(self._target_resnet.trainable_weights, resnet_weights),))

    if self._use_momentum_proj_head:
      head_weights = [
          self._projection_head.trainable_weights[idx]
          for idx in self._head_weight_permutation
      ]
      ctx.merge_call(_update, args=(
          zip(self._target_head.trainable_weights, head_weights),))

  def __call__(self, inputs, training):
    target_proj_output = None
    z_mu = None
    if tf.is_tensor(inputs):
      inputs = [inputs]
    features_list = inputs
    if self._target_resnet is None:
      # SimCLR
      features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)
      # Base network forward pass.
      hiddens = self._resnet(features, training=training)
      projection_head_outputs, supervised_head_inputs = self._projection_head(
          hiddens, training)
    else:
      # BYOL online path
      hidden_1 = self._resnet(features_list[0], training=training)
      projection_head_1, supervised_head_inputs = self._projection_head(
          hidden_1, training)
      if self._use_projector:
        projection_head_1, z_mu = self._projector(projection_head_1, training)

      # BYOL target path
      if training and FLAGS.train_mode == 'pretrain' and (
          not self.is_export_single_branch):
        # Only when training is true and train_mode == 'pretrain', we expect
        # feature_list to contain 2 views.
        hidden_2 = self._target_resnet(features_list[1], training=training)
        hidden_2 = tf.stop_gradient(hidden_2)
        projection_head_2, _ = self._target_head(hidden_2, training=training)
        if self._use_momentum_proj_head:
          projection_head_2 = tf.stop_gradient(projection_head_2)
        projection_head_outputs = tf.concat(
            [projection_head_1, projection_head_2], axis=0)
        if self._use_target_projector:
          target_proj_output, _ = self._target_projector(
              tf.math.l2_normalize(projection_head_2, -1),
              training=training)
      else:
        projection_head_outputs = projection_head_1

    if FLAGS.mode == 'extract_features':
      return supervised_head_inputs
    elif FLAGS.train_mode == 'finetune':
      supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                     training)
      return None, supervised_head_outputs, None, None, None, None, None, None
    elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
      # When performing pretraining and linear evaluation together we do not
      # want information from linear eval flowing back into pretraining network
      # so we put a stop_gradient.
      supervised_head_outputs = self.supervised_head(
          tf.stop_gradient(supervised_head_inputs), training)
      return projection_head_outputs, supervised_head_outputs, target_proj_output, z_mu
    else:
      return projection_head_outputs, None, target_proj_output, z_mu


def build_model(num_classes):
  """Returns the model."""
  resnet_model = resnet.resnet(
      resnet_depth=FLAGS.resnet_depth,
      width_multiplier=FLAGS.width_multiplier,
      cifar_stem=FLAGS.image_size <= 32,
      name='resnet')
  target_encoder = None
  if FLAGS.use_momentum_encoder:
    target_encoder = resnet.resnet(
        resnet_depth=FLAGS.resnet_depth,
        width_multiplier=FLAGS.width_multiplier,
        cifar_stem=FLAGS.image_size <= 32,
        trainable=True,
        name='resnet_ema')
  model = Model(resnet_model,
                num_classes,
                target_encoder=target_encoder,
                use_momentum_proj_head=FLAGS.use_momentum_proj_head,
                use_projector=FLAGS.use_projector,
                use_target_projector=FLAGS.use_target_projector)
  return model


def build_optimizer(learning_rate):
  """Returns the optimizer."""
  if FLAGS.optimizer == 'momentum':
    return tf.keras.optimizers.SGD(learning_rate, FLAGS.momentum, nesterov=True)
  elif FLAGS.optimizer == 'adam':
    return tf.keras.optimizers.Adam(learning_rate)
  elif FLAGS.optimizer == 'lars':
    return lars_optimizer.LARSOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        exclude_from_weight_decay=[
            'batch_normalization', 'bias', 'head_supervised', 'resnet_ema',
            'target_head_ema'
        ])
  else:
    raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))


def image_postprocess(inputs, training):
  """Processing views."""
  features = inputs
  if training and FLAGS.train_mode == 'pretrain':
    num_transforms = 2
    if FLAGS.fine_tune_after_block > -1:
      raise ValueError('Does not support layer freezing during pretraining,'
                       'should set fine_tune_after_block<=-1 for safety.')
  else:
    num_transforms = 1

  # Split channels, and optionally apply extra batched augmentation.
  # [B, H, W, Cx2] -> [[B, H, W, C], [B, H, W, C]]
  features_list = tf.split(
      features, num_or_size_splits=num_transforms, axis=-1)
  if FLAGS.no_augmentation:
    blur_probabilities = [0.0]  # expecting only one view
    solarization_probabilities = [0.0]  # expecting only one view
    features_list = data_util.batch_augmentation_byol(
        features_list,
        FLAGS.image_size,
        FLAGS.image_size,
        blur_probabilities=blur_probabilities,
        solarization_probabilities=solarization_probabilities)
  elif FLAGS.batch_image_augmentation_type == 'simclr':
    if FLAGS.use_blur and training and FLAGS.train_mode == 'pretrain':
      features_list = data_util.batch_augmentation_simclr(features_list,
                                                          FLAGS.image_size,
                                                          FLAGS.image_size)
    elif FLAGS.use_blur and training and (
        FLAGS.train_mode == 'finetune' and FLAGS.ft_with_full_aug):
      features_list = data_util.batch_augmentation_simclr(features_list,
                                                          FLAGS.image_size,
                                                          FLAGS.image_size)
  elif FLAGS.batch_image_augmentation_type == 'byol':
    if training and FLAGS.train_mode == 'pretrain':
      blur_probabilities = [1.0, 0.1]
      solarization_probabilities = [0.0, 0.2]
    elif training and FLAGS.train_mode == 'finetune' and FLAGS.ft_with_full_aug:
      blur_probabilities = [1.0]
      solarization_probabilities = [0.0]
    else:
      blur_probabilities = [0.0]  # expecting only one view
      solarization_probabilities = [0.0]  # expecting only one view
    features_list = data_util.batch_augmentation_byol(
        features_list,
        FLAGS.image_size,
        FLAGS.image_size,
        blur_probabilities=blur_probabilities,
        solarization_probabilities=solarization_probabilities)
  else:
    raise NotImplementedError
  return features_list
