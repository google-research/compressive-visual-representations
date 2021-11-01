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
"""Contrastive loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

FLAGS = flags.FLAGS
LARGE_NUM = 1e9


def tpu_cross_replica_concat(tensor, strategy=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    strategy: a `tf.distribute.Strategy`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if strategy is None or strategy.num_replicas_in_sync <= 1:
    return tensor

  num_replicas = strategy.num_replicas_in_sync

  replica_context = tf.distribute.get_replica_context()
  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[replica_context.replica_id_in_sync_group]],
        updates=[tensor],
        shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                            ext_tensor)
    # full data: [1, 2, 3, 4]
    # 2 replicas
    # replica 1: [1, 2, 0, 0]
    #         2: [0, 0, 3, 4]
    # all_reduce + SUM:
    #            [1, 2, 3, 4]
    # batch size 512
    # each replica: 256

    # replica 1 batch size: 256
    # replica 2 batch size: 16

    # Note that each shard does not necessarily has same shared batch size (e.g.
    # when dataset is not repeated, there could be an incomplete batch at the
    # end). MirrorStrategy could raise the error but TPUStrategy will pass but
    # create incorrect results.

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])


def add_supervised_loss(labels, logits):
  """Compute mean supervised loss over local batch."""
  if FLAGS.use_binary_cross_entropy:
    losses = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                    logits)
    # Scenic code sums up each label if I remember. So let's check that
    losses = tf.reduce_sum(losses, axis=-1)
  else:
    losses = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                    logits)
  return tf.reduce_mean(losses)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=0.1,
                         bidirectional=True,
                         double_batch_trick=True,
                         strategy=None):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (B, dim).
    hidden_norm: whether to l2-normalize the hidden vector.
    temperature: temperature scaling.
    bidirectional: whether to compute loss in a bidirectional manner.
    double_batch_trick: whether to use negatives from forward encoder e to
      double the batch size.
    strategy: a `tf.distribute.Strategy`.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    A metrics dictionary.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if strategy is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    replica_context = tf.distribute.get_replica_context()
    replica_id = tf.cast(
        tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
    if double_batch_trick:
      labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
      mi_upper_bound = tf.math.log(tf.cast(enlarged_batch_size*2-1, tf.float32))
    else:
      labels = tf.one_hot(labels_idx, enlarged_batch_size)
      mi_upper_bound = tf.math.log(tf.cast(enlarged_batch_size, tf.float32))
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    if double_batch_trick:
      labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
      mi_upper_bound = tf.math.log(tf.cast(batch_size*2-1, tf.float32))
    else:
      labels = tf.one_hot(tf.range(batch_size), batch_size)
      mi_upper_bound = tf.math.log(tf.cast(batch_size, tf.float32))

  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  if double_batch_trick:
    logits_aa = tf.matmul(
        hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_ab = tf.concat([logits_ab, logits_aa], 1)

  loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits_ab)
  i_y_zx = mi_upper_bound - loss
  metrics = {}
  metrics['i_y_zx'] = tf.reduce_mean(i_y_zx)

  if bidirectional:
    logits_ba = tf.matmul(
        hidden2, hidden1_large, transpose_b=True) / temperature
    if double_batch_trick:
      logits_bb = tf.matmul(
          hidden2, hidden2_large, transpose_b=True) / temperature
      logits_bb = logits_bb - masks * LARGE_NUM
      logits_ba = tf.concat([logits_ba, logits_bb], 1)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, logits_ba)
    i_x_zy = mi_upper_bound - loss_b
    metrics['i_x_zy'] = tf.reduce_mean(i_x_zy)
    loss = loss + loss_b

  loss = tf.reduce_mean(loss)
  return loss, logits_ab, labels, metrics


def add_contrastive_ceb_loss(hidden,
                             kappa_e=1024.0,
                             kappa_b=10.0,
                             beta=1.0,
                             bidirectional=True,
                             sampling=True,
                             double_batch_trick=True,
                             strategy=None):
  """Compute contrastive version of CEB loss (von Mises-Fisher) for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (B, dim).
    kappa_e: forward encoder concentration.
    kappa_b: backward encoder concentration.
    beta: CEB beta.
    bidirectional: whether to compute loss in a bidirectional manner.
    sampling: whether sampling from forward encoder or not.
    double_batch_trick: whether to use negatives from encoder e to
      double the batch size.
    strategy: a `tf.distribute.Strategy`.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    A metrics dictionary.
  """
  # Get (normalized) hidden1 and hidden2.
  hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if strategy is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)

    enlarged_batch_size = tf.shape(hidden1_large)[0]
    replica_context = tf.distribute.get_replica_context()
    replica_id = tf.cast(
        tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
    # assuming uniform sampling, MI upper bound == H(X) == H(Y) == -log 1/K
    if double_batch_trick:
      labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
      mi_upper_bound = tf.math.log(tf.cast(enlarged_batch_size*2-1, tf.float32))
    else:
      labels = tf.one_hot(labels_idx, enlarged_batch_size)
      mi_upper_bound = tf.math.log(tf.cast(enlarged_batch_size, tf.float32))
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels_idx = tf.range(batch_size)
    masks = tf.one_hot(labels_idx, batch_size)
    if double_batch_trick:
      labels = tf.one_hot(labels_idx, batch_size * 2)
      mi_upper_bound = tf.math.log(tf.cast(batch_size*2-1, tf.float32))
    else:
      labels = tf.one_hot(labels_idx, batch_size)
      mi_upper_bound = tf.math.log(tf.cast(batch_size, tf.float32))

  # e_zx: [B, (Z)]
  e_zx = tfd.VonMisesFisher(hidden1, kappa_e)  # scale needs to be batched
  b_zy = tfd.VonMisesFisher(hidden2, kappa_b)
  # b_zy_large: [Bex, (Z)]
  b_zy_large = tfd.VonMisesFisher(hidden2_large, kappa_b)

  # Reversed distributions for bidirectional learning and additional negatives
  if bidirectional or double_batch_trick:
    # [B, (Z)]
    e2_zx = tfd.VonMisesFisher(hidden1, kappa_b)  # scale needs to be batched
    b2_zy = tfd.VonMisesFisher(hidden2, kappa_e)
    # [Bex, (Z)]
    e2_zx_large = tfd.VonMisesFisher(hidden1_large, kappa_b)

  metrics = {}
  # X -> Y
  # zx: [B, Z]
  if sampling:
    zx = e_zx.sample()
  else:
    zx = e_zx.mean_direction
  log_e_zx_x = e_zx.log_prob(zx)
  log_b_zx_y = b_zy.log_prob(zx)
  i_xzx_y = log_e_zx_x - log_b_zx_y  # [B,], residual information I(X;Z|Y)
  # logits_ab: [B, Bex], zx -> [B, 1, Z]
  logits_ab = b_zy_large.log_prob(zx[:, None, :])
  if double_batch_trick:
    # logits_aa: [B, Bex], zx -> [B, 1, Z]
    logits_aa = e2_zx_large.log_prob(zx[:, None, :])
    logits_aa = logits_aa - masks * LARGE_NUM  # Mask out itself
    logits_ab = tf.concat([logits_ab, logits_aa], -1)

  # original_loss_a = -log p(y|zx) -> H(Y|Zx)
  cat_dist_ab = tfd.Categorical(logits=logits_ab)
  h_y_zx = -cat_dist_ab.log_prob(labels_idx)
  i_y_zx = mi_upper_bound - h_y_zx
  loss = beta * i_xzx_y - i_y_zx
  metrics['i_xzx_y'] = tf.reduce_mean(i_xzx_y)
  metrics['i_y_zx'] = tf.reduce_mean(i_y_zx)
  metrics['h_e_zx_x'] = tf.reduce_mean(-log_e_zx_x)
  metrics['h_b_zx_y'] = tf.reduce_mean(-log_b_zx_y)

  # Y -> X
  if bidirectional:
    if sampling:
      zy = b2_zy.sample()
    else:
      zy = b2_zy.mean_direction
    log_b2_zy_y = b2_zy.log_prob(zy)
    log_e2_zy_x = e2_zx.log_prob(zy)
    i_yzy_x = log_b2_zy_y - log_e2_zy_x  # [B,], residual information I(Y;Z|X)

    # logits_ba: [B, Bex], zy -> [B, 1, Z]
    logits_ba = e2_zx_large.log_prob(zy[:, None, :])
    if double_batch_trick:
      # logits_bb: [B, Bex], zy -> [B, 1, Z]
      logits_bb = b_zy_large.log_prob(zy[:, None, :])
      logits_bb = logits_bb - masks * LARGE_NUM  # Mask out itself
      logits_ba = tf.concat([logits_ba, logits_bb], -1)

    # original_loss_b = -log p(x|zy) -> H(X|Zy)
    cat_dist_ba = tfd.Categorical(logits=logits_ba)
    h_x_zy = -cat_dist_ba.log_prob(labels_idx)
    i_x_zy = mi_upper_bound - h_x_zy
    loss += beta * i_yzy_x - i_x_zy
    metrics['i_yzy_x'] = tf.reduce_mean(i_yzy_x)
    metrics['i_x_zy'] = tf.reduce_mean(i_x_zy)

  loss = tf.reduce_mean(loss)
  return loss, logits_ab, labels, metrics


def add_byol_loss(hidden, hidden_norm=True, byol_loss_weight=1.0):
  """Compute the BYOL loss.

  Args:
    hidden: hidden vector (`Tensor`) of shape (B, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    byol_loss_weight: BYOL loss weight.

  Returns:
    A loss scalar.
    A metrics dictionary.
  """
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  loss = tf.reduce_sum(tf.math.square(hidden1 - hidden2), axis=-1)
  loss = tf.reduce_mean(loss)
  metrics = {}
  metrics['byol_loss'] = loss
  loss = loss * byol_loss_weight
  return loss, metrics


def add_byol_ceb_loss(hidden,
                      hidden2,
                      target_proj_output,
                      target_proj_output2,
                      z_mu,
                      z_mu2,
                      kappa_e=1024.0,
                      kappa_b=10.0,
                      beta=1.0,
                      byol_loss_weight=1.0):
  """Compute the bidirectional BYOL CEB loss.

  Args:
    hidden: hidden vector (`Tensor`) of shape (B, dim).
    hidden2: hidden vector (`Tensor`) of shape (B, dim).
    target_proj_output: hidden vector (`Tensor`) of shape (B, dim).
    target_proj_output2: hidden vector (`Tensor`) of shape (B, dim).
    z_mu: a concatenation of a sample z from e(z|x) and the mean of e(z|x).
    z_mu2: a concatenation of a sample z from e(z|x) and the mean of e(z|x), in
      the reversed direction.
    kappa_e: forward encoder concentration.
    kappa_b: backward encoder concentration.
    beta: CEB beta.
    byol_loss_weight: BYOL loss weight, equivalent to kappa_d/2.

  Returns:
    A loss scalar.
    A metrics dictionary.
  """
  hidden = tf.math.l2_normalize(hidden, -1)
  hidden2 = tf.math.l2_normalize(hidden2, -1)
  hidden_online1, hidden_target1 = tf.split(hidden, 2, 0)
  hidden_online2, hidden_target2 = tf.split(hidden2, 2, 0)

  # BYOL regression loss (view 1, 2)
  byol_loss = tf.reduce_sum(
      tf.math.square(hidden_online1 - hidden_target1), axis=-1)
  # BYOL regression loss in the reversed directionfor (view 2,1)
  byol_loss2 = tf.reduce_sum(
      tf.math.square(hidden_online2 - hidden_target2), axis=-1)

  metrics = {}
  metrics['byol_loss'] = tf.reduce_mean(byol_loss)
  loss = (byol_loss + byol_loss2) * byol_loss_weight

  # e_zx: [B, (Z)]
  z1, mu1 = tf.split(z_mu, 2, -1)
  z2, mu2 = tf.split(z_mu2, 2, -1)
  e_zx1 = tfd.VonMisesFisher(mu1, kappa_e)
  e_zx2 = tfd.VonMisesFisher(mu2, kappa_e)
  target_proj_output = tf.math.l2_normalize(target_proj_output, -1)
  target_proj_output2 = tf.math.l2_normalize(target_proj_output2, -1)
  b_zy1 = tfd.VonMisesFisher(target_proj_output2, kappa_b)
  b_zy2 = tfd.VonMisesFisher(target_proj_output, kappa_b)

  # z: [B, Z]
  log_e_zx1 = e_zx1.log_prob(z1)
  log_b_zy1 = b_zy1.log_prob(z1)
  log_e_zx2 = e_zx2.log_prob(z2)
  log_b_zy2 = b_zy2.log_prob(z2)
  i_xzy1 = log_e_zx1 - log_b_zy1  # [B,]
  i_xzy2 = log_e_zx2 - log_b_zy2  # [B,]

  metrics['h_b_zx_y'] = tf.reduce_mean(-log_b_zy1)
  metrics['h_e_zx_x'] = tf.reduce_mean(-log_e_zx1)
  metrics['i_xzx_y'] = tf.reduce_mean(i_xzy1)

  loss += beta * (i_xzy1 + i_xzy2)
  loss = tf.reduce_mean(loss)

  return loss, metrics
