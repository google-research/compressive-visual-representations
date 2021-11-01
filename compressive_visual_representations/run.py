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
"""The main training pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging

from compressive_visual_representations import data as data_lib
from compressive_visual_representations import metrics
from compressive_visual_representations import model as model_lib
from compressive_visual_representations import model_util
from compressive_visual_representations import objective as obj_lib
from compressive_visual_representations import resnet

from flax.metrics import tensorboard

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub  # pylint: disable=unused-import
import tensorflow_probability as tfp

tfd = tfp.distributions

FLAGS = flags.FLAGS

# Optimization
flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_enum(
    'learning_rate_schedule', 'cosine', ['cosine', 'constant'],
    'Learning rate schedule.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_float(
    'ema_alpha', 0.999,
    'Moving average co-efficient to use for the ema_encoder, if applicable'
)

# Data
flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'train_summary_steps', 100,
    'Steps before saving training summaries. If 0, will not save.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'validation',
    'Split for evaluation.')

flags.DEFINE_string(
    'dataset', 'imagenet2012',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

# Program config
flags.DEFINE_enum(
    'mode', 'train_then_eval', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', True,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_bool(
    'reinit_logits_layer', True,
    'If True, re-initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linera head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_bool(
    'use_tpu', True,
    'Whether to run on TPU.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_boolean(
    'export_eval_model', True,
    'Whether to export tf.SavedModel of evaluation graph')

flags.DEFINE_boolean(
    'export_train_model', True,
    'Whether to export tf.SavedModel of evaluation graph')

flags.DEFINE_boolean(
    'explicitly_watch_vars', False,
    'Whether or not to watch trainable variables explicitly.')

# Losses
flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_string(
    'pretrain_loss', 'contrastive',
    'Name for pretrain loss.')

flags.DEFINE_boolean(
    'ceb_sampling', True,
    'sample z or use the mean')

flags.DEFINE_float('ceb_beta', 0.0, 'Beta that controls compression.')

flags.DEFINE_float('kappa_e', 1024.0, 'Forward encoder vMF concentration')

flags.DEFINE_float('kappa_b', 10.0, 'Backward encoder vMF concentration')

flags.DEFINE_float('byol_loss_weight', 1.0, 'BYOL loss weight, 2x kappa_d')

flags.DEFINE_boolean(
    'double_batch_trick', True,
    'Whether or not to use negative examples from same encoder.')

flags.DEFINE_string(
    'beta_schedule', 'constant',
    'constant | linear_warmup | cosine')

# Architecture
flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 256,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 2,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', -1,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'use_projector', False,
    'Whether or not to use projector (BYOL)')

flags.DEFINE_boolean(
    'stochastic_projector', False,
    'Use stochastic projector (C-BYOL).')

flags.DEFINE_boolean(
    'use_target_projector', False,
    'Whether or not to use target projector (C-BYOL)')

flags.DEFINE_boolean(
    'projector_mid_bn', True,
    'Whether or not to use batch norm in middle layers.')

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_string(
    'mlp_init', 'byol', 'simclr|byol')

flags.DEFINE_boolean(
    'proj_head_output_bn', False,
    'Whether or not to use batch norm on the hidden vector.')

flags.DEFINE_integer(
    'proj_head_hidden_dim', 4096,
    'Projection head hidden dimension.')

flags.DEFINE_boolean(
    'use_momentum_encoder', False,
    'Whether or not to use an EMA of the model weights for the encoder')

flags.DEFINE_boolean(
    'use_momentum_proj_head', False,
    'Whether or not to make the target projection head EMA')

# Augmentations
flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')

flags.DEFINE_float(
    'color_jitter_brightness', 0.4,
    'Brightness adjustment max intensity.')

flags.DEFINE_float(
    'color_jitter_contrast', 0.4,
    'aturation adjustment max intensity.')

flags.DEFINE_float(
    'color_jitter_saturation', 0.2,
    'Saturation adjustment max intensity.')

flags.DEFINE_float(
    'color_jitter_hue', 0.1,
    'Hue adjustment max intensity.')

flags.DEFINE_string(
    'batch_image_augmentation_type', 'byol', 'simclr|byol')

flags.DEFINE_string(
    'bbox_sampling', 'logarithmic', 'uniform (simclr) | logarithmic (byol)')

flags.DEFINE_string(
    'brightness_fn', 'simclrv1', '\'simclrv1\' (byol) or \'simclrv2\'')

flags.DEFINE_boolean(
    'clip_by_value_after_resizing', True,
    'Whether or not to clip pixel values to [0, 1] after resizing (byol).')

flags.DEFINE_boolean(
    'clip_by_value_after_grayscale', True,
    'Whether or not to clip pixel values to [0, 1] after grayscale (byol).')

flags.DEFINE_float(
    'area_range_lower_bound', 0.08,
    'Area range lower bound for cropping')

flags.DEFINE_boolean(
    'no_augmentation', False,
    'Whether to disable all augmentation and simply rescale images')

flags.DEFINE_boolean(
    'ft_with_full_aug', False,
    'Whether to finetune with full augmentation.')

# TODO(leekh, aarnab): figure out a better way to rename classification head.
flags.DEFINE_boolean(
    'rename_supervised_head', False,
    'If true, rename supervised head. Only valid if finetuning')

# Used in objective.add_supervised_loss
flags.DEFINE_boolean(
    'use_binary_cross_entropy', False,
    'If true, use binary cross entropy instead of softmax cross entropy for'
    'supervised loss.')


class SummaryWriter(tensorboard.SummaryWriter):
  """Summary writer object."""

  def __init__(self, log_dir, pool):
    del pool
    super().__init__(log_dir)

  def scalar(self, tag, value, step):
    """Writes scalar summary."""

    super().scalar(tag, value, step)


def get_salient_tensors_dict(include_projection_head):
  """Returns a dictionary of tensors."""
  graph = tf.compat.v1.get_default_graph()
  result = {}
  for i in range(1, 5):
    result['block_group%d' % i] = graph.get_tensor_by_name(
        'resnet/block_group%d/block_group%d:0' % (i, i))
  # Comment out these for backward compatibility
  result['initial_conv'] = graph.get_tensor_by_name(
      'resnet/initial_conv/Identity:0')
  result['initial_max_pool'] = graph.get_tensor_by_name(
      'resnet/initial_max_pool/Identity:0')
  result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
  result['logits_sup'] = graph.get_tensor_by_name(
      'head_supervised/logits_sup:0')
  result['supervised_head_input'] = graph.get_tensor_by_name(
      'projection_head/supervised_head_input:0')
  if include_projection_head:
    result['proj_head_input'] = graph.get_tensor_by_name(
        'projection_head/proj_head_input:0')
    result['proj_head_output'] = graph.get_tensor_by_name(
        'projection_head/proj_head_output:0')

  return result


def build_saved_model(model, include_projection_head=True):
  """Returns a tf.Module for saving to SavedModel."""

  class SSLModel(tf.Module):
    """Saved model for exporting to hub."""

    def __init__(self, model):
      self.model = model
      # This can't be called `trainable_variables` because `tf.Module` has
      # a getter with the same name.
      self.trainable_variables_list = model.trainable_variables
      self.variables_list = model.variables

    @tf.function
    def __call__(self, inputs, trainable):
      # Normally, training=True, means that it splits from 6 channels to
      # two 3 channel images
      self.model(inputs, training=trainable)
      return get_salient_tensors_dict(include_projection_head)

  module = SSLModel(model)
  input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)

  module.__call__.get_concrete_function(input_spec, trainable=True)
  module.__call__.get_concrete_function(input_spec, trainable=False)
  return module


def save(model, global_step, suffix=''):
  """Export as SavedModel for finetuning and inference."""
  model.is_export_single_branch = True
  saved_model = build_saved_model(model)
  export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
  if suffix:
    export_dir = os.path.join(export_dir, suffix)
  checkpoint_export_dir = os.path.join(export_dir, str(global_step))
  if tf.io.gfile.exists(checkpoint_export_dir):
    tf.io.gfile.rmtree(checkpoint_export_dir)
  tf.saved_model.save(saved_model, checkpoint_export_dir)

  if FLAGS.keep_hub_module_max > 0:
    # Delete old exported SavedModels.
    exported_steps = []
    for subdir in tf.io.gfile.listdir(export_dir):
      if not subdir.isdigit():
        continue
      exported_steps.append(int(subdir))
    exported_steps.sort()
    for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
      tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))
  model.is_export_single_branch = False


def is_pretraining():
  return FLAGS.train_mode == 'pretrain'


def build_checkpoint(model, global_step, optimizer=None):
  """Returns a tf.train.Checkpoint."""
  # We do not want to load the pretraining optimizer and pretrain global_step
  # when finetuning so we use different names for those than the pretrain
  # Checkpoint. The model has different names for supervised and constrastive
  # heads so those are fine.
  checkpoint = tf.train.Checkpoint(model=model)
  global_step_attr = ('global_step'
                      if is_pretraining() else 'finetune_global_step')
  setattr(checkpoint, global_step_attr, global_step)
  if optimizer is not None:
    optimizer_attr = 'optimizer' if is_pretraining() else 'finetune_optimizer'
    setattr(checkpoint, optimizer_attr, optimizer)
  return checkpoint


def checkpoint_get_global_step(checkpoint):
  global_step_attr = ('global_step'
                      if is_pretraining() else 'finetune_global_step')
  return getattr(checkpoint, global_step_attr)


def json_serializable(val):
  try:
    json.dumps(val)
    return True
  except TypeError:
    return False


def perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology):
  """Perform evaluation."""
  # Build input pipeline.
  ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, False,
                                          strategy, topology)
  pool = multiprocessing.pool.ThreadPool()
  summary_writer = SummaryWriter(FLAGS.model_dir, pool)

  # Build metrics.
  with strategy.scope():
    all_metrics = []
    regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
    all_metrics.append(regularization_loss)
    if FLAGS.train_mode == 'pretrain':
      contrast_loss = tf.keras.metrics.Mean('eval/ssl_contrast_loss')
      contrastive_top_1_accuracy = tf.keras.metrics.Accuracy(
          'eval/ssL_contrastive_top_1_accuracy')
      contrastive_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
          5, 'eval/ssl_contrastive_top_5_accuracy')
      all_metrics.extend([
          contrast_loss, contrastive_top_1_accuracy, contrastive_top_5_accuracy
      ])
    if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
      label_top_1_accuracy = tf.keras.metrics.Accuracy(
          'eval/label_top_1_accuracy')
      label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
          5, 'eval/label_top_5_accuracy')
      all_metrics.extend([label_top_1_accuracy, label_top_5_accuracy])

  # Restore checkpoint.
  checkpoint = build_checkpoint(model, tf.Variable(0, dtype=tf.int64))
  checkpoint.restore(ckpt).expect_partial()

  # A keras work-around for fine-tuning on a different dataset.
  if FLAGS.rename_supervised_head:
    model.reset_heads()

  logging.info('Restoring from %s', ckpt)
  global_step = checkpoint_get_global_step(checkpoint)
  logging.info('Performing eval at step %d', global_step.numpy())

  def single_step(features, labels):
    features = model_lib.image_postprocess(features, training=False)
    _, supervised_head_outputs, _, _ = model(features, training=False)
    assert supervised_head_outputs is not None
    l = labels['labels']
    metrics.update_finetune_metrics_eval(label_top_1_accuracy,
                                         label_top_5_accuracy,
                                         supervised_head_outputs, l)
    reg_loss = model_util.add_weight_decay(model, adjust_per_optimizer=True)
    regularization_loss.update_state(reg_loss)

  with strategy.scope():

    @tf.function
    def run_single_step(iterator):
      images, labels = next(iterator)
      features, labels = images, {'labels': labels}
      strategy.run(single_step, (features, labels))

    iterator = iter(ds)
    for i in range(eval_steps):
      run_single_step(iterator)
      logging.info('Completed eval for %d / %d steps', i + 1, eval_steps)
    logging.info('Finished eval for %s', ckpt)

  # Write summaries
  cur_step = global_step.numpy()
  logging.info('Writing summaries for %d step', cur_step)
  metrics.log_and_write_metrics(all_metrics, cur_step, summary_writer)
  summary_writer.flush()

  # Record results as JSON.
  result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
  result = {metric.name: metric.result().numpy() for metric in all_metrics}
  result['global_step'] = checkpoint_get_global_step(checkpoint).numpy()
  logging.info(result)
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
      FLAGS.model_dir, 'result_%d.json'%result['global_step'])
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    serializable_flags = {}
    for key, val in FLAGS.flag_values_dict().items():
      # Some flag value types e.g. datetime.timedelta are not json serializable,
      # filter those out.
      if json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f)

  if FLAGS.export_eval_model and (
      FLAGS.train_mode == 'pretrain' or FLAGS.train_mode == 'finetune'):
    save(model, global_step=result['global_step'], suffix='eval')

  pool.close()
  pool.join()
  return result


def _restore_latest_or_from_pretrain(checkpoint_manager):
  """Restores the latest ckpt if training already.

  Or restores from FLAGS.checkpoint if in finetune mode.

  Args:
    checkpoint_manager: tf.train.CheckpointManager.

  Returns:
    True if restored from checkpoint
  """
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # The model is not build yet so some variables may not be available in
    # the object graph. Those are lazily initialized. To suppress the warning
    # in that case we specify `expect_partial`.
    logging.info('Restoring from %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
    return True
  elif FLAGS.train_mode == 'finetune':
    # if training supervised models from scratch, return False here
    if FLAGS.reinit_logits_layer:
      model = checkpoint_manager.checkpoint.model
      output_layer_parameters = model.supervised_head.get_weights()
    # Restore from pretrain checkpoint.
    assert FLAGS.checkpoint, 'Missing pretrain checkpoint.'
    logging.info('Restoring from %s', FLAGS.checkpoint)
    checkpoint_manager.checkpoint.restore(FLAGS.checkpoint).expect_partial()
    if FLAGS.reinit_logits_layer:
      model.supervised_head.set_weights(output_layer_parameters)

    if FLAGS.rename_supervised_head:
      model.reset_heads()
    return True
  return False


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL

  builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
  builder.download_and_prepare()
  num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
  num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
  num_classes = builder.info.features['label'].num_classes

  train_steps = model_util.get_train_steps(num_train_examples)
  eval_steps = int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))
  epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

  logging.info('# train examples: %d', num_train_examples)
  logging.info('# train_steps: %d', train_steps)
  logging.info('# eval examples: %d', num_eval_examples)
  logging.info('# eval steps: %d', eval_steps)

  resnet.BATCH_NORM_DECAY = FLAGS.batch_norm_decay

  checkpoint_steps = (
      FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))

  topology = None
  if FLAGS.use_tpu:
    if FLAGS.tpu_name:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.master)
    tf.config.experimental_connect_to_cluster(cluster)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster)
    logging.info('Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(cluster)
  else:
    strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = model_lib.build_model(num_classes)

    if FLAGS.train_mode == 'pretrain' and not model.is_ema_initialised():
      logging.info('Call initialise_ema()')
      model.initialise_ema(
          input_shape=[1, FLAGS.image_size, FLAGS.image_size, 3])
    elif FLAGS.train_mode == 'finetune':
      model.initialise_networks(
          input_shape=[1, FLAGS.image_size, FLAGS.image_size, 3])

  if FLAGS.mode == 'eval':
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, min_interval_secs=15):
      result = perform_evaluation(model, builder, eval_steps, ckpt, strategy,
                                  topology)
      if result['global_step'] >= train_steps:
        logging.info('Eval complete. Exiting...')
        return
  else:
    pool = multiprocessing.pool.ThreadPool()
    summary_writer = SummaryWriter(FLAGS.model_dir, pool)

    with strategy.scope():
      # Build input pipeline.
      ds = data_lib.build_distributed_dataset(builder, FLAGS.train_batch_size,
                                              True, strategy, topology)

      # Build LR schedule and optimizer.
      if FLAGS.learning_rate_schedule == 'cosine':
        learning_rate = model_util.WarmUpAndCosineDecay(FLAGS.learning_rate,
                                                        num_train_examples)
      else:
        if FLAGS.learning_rate_scaling == 'linear':
          scaled_lr = FLAGS.learning_rate * FLAGS.train_batch_size / 256.
        elif FLAGS.learning_rate_scaling == 'sqrt':
          scaled_lr = FLAGS.learning_rate * math.sqrt(FLAGS.train_batch_size)
        else:
          raise ValueError('Unknown learning rate scaling {}'.format(
              FLAGS.learning_rate_scaling))
        learning_rate = scaled_lr
      optimizer = model_lib.build_optimizer(learning_rate)

      # Build EMA schedule
      ema_alpha_fn = model_util.cosine_ramping_schedule(FLAGS.ema_alpha,
                                                        num_train_examples)

      # beta schedule
      if FLAGS.beta_schedule == 'constant':
        beta_fn = lambda x: FLAGS.ceb_beta
      elif FLAGS.beta_schedule == 'linear_warmup':
        warmup_steps = int(
            round(FLAGS.warmup_epochs * num_train_examples //
                  FLAGS.train_batch_size))
        beta_fn = model_util.warmup_schedule(FLAGS.ceb_beta,
                                             warmup_steps=warmup_steps)
      elif FLAGS.beta_schedule == 'cosine':
        beta_fn = model_util.cosine_ramping_from_zero_schedule(
            FLAGS.ceb_beta,
            num_train_examples)
      else:
        raise NotImplementedError

      # Build metrics.
      all_metrics = []  # For summaries.
      weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
      total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
      all_metrics.extend([weight_decay_metric, total_loss_metric])
      if FLAGS.train_mode == 'pretrain':
        ema_alpha_metric = tf.keras.metrics.Mean('train/ema_alpha_log')
        ceb_beta_metric = tf.keras.metrics.Mean('train/ceb_beta_log')
        contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
        contrast_entropy_metric = tf.keras.metrics.Mean(
            'train/contrast_entropy')
        i_xzx_y_metric = tf.keras.metrics.Mean('train/i_xzx_y')
        i_y_zx_metric = tf.keras.metrics.Mean('train/i_y_zx')
        h_e_zx_x_metric = tf.keras.metrics.Mean('train/h_e_zx_x')
        h_b_zx_y_metric = tf.keras.metrics.Mean('train/h_b_zx_y')
        i_yzy_x_metric = tf.keras.metrics.Mean('train/i_yzy_x')
        i_x_zy_metric = tf.keras.metrics.Mean('train/i_x_zy')
        byol_loss_metric = tf.keras.metrics.Mean('train/byol_loss')
        all_metrics.extend([
            contrast_loss_metric, contrast_acc_metric,
            contrast_entropy_metric, ema_alpha_metric, ceb_beta_metric,
            i_xzx_y_metric, i_y_zx_metric, h_e_zx_x_metric, h_b_zx_y_metric,
            i_yzy_x_metric, i_x_zy_metric, byol_loss_metric,
        ])

      if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        all_metrics.extend([supervised_loss_metric, supervised_acc_metric])

    # Restore checkpoint if available.
    checkpoint = build_checkpoint(model, optimizer.iterations, optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    is_restored_from_checkpoint = _restore_latest_or_from_pretrain(
        checkpoint_manager)
    logging.info('Is restored from checkpoint? %s',
                 str(is_restored_from_checkpoint))

    def single_step(features, labels):
      global_step = tf.cast(optimizer.iterations, dtype=tf.float32)
      if FLAGS.explicitly_watch_vars:  # for BYOL
        trainable_variables = model._resnet.trainable_variables  # pylint: disable=protected-access
        trainable_variables += model._projection_head.trainable_variables  # pylint: disable=protected-access
        if FLAGS.use_momentum_encoder and (not FLAGS.use_momentum_proj_head):
          trainable_variables += model._target_head.trainable_variables  # pylint: disable=protected-access
        if FLAGS.use_projector:
          trainable_variables += model._projector.trainable_variables  # pylint: disable=protected-access
        if FLAGS.use_target_projector:
          trainable_variables += model._target_projector.trainable_variables  # pylint: disable=protected-access
        trainable_variables += model.supervised_head.trainable_variables
      with tf.GradientTape(watch_accessed_variables=(
          not FLAGS.explicitly_watch_vars)) as tape:
        if FLAGS.explicitly_watch_vars:
          tape.watch(trainable_variables)
        features = model_lib.image_postprocess(features, training=True)
        proj_outputs, supervised_head_outputs, target_proj_output, z_mu = model(
            features, training=True)
        if FLAGS.pretrain_loss in ['byol', 'byol_ceb']:
          features2 = [features[1], features[0]]
          proj_outputs2, _, target_proj_output2, z_mu2 = model(
              features2, training=True)
        loss = None
        output_metrics = {}
        ceb_beta = beta_fn(global_step)
        if proj_outputs is not None:
          if FLAGS.pretrain_loss == 'contrastive':
            con_loss, logits_con, labels_con, output_metrics = obj_lib.add_contrastive_loss(
                proj_outputs,
                temperature=FLAGS.temperature,
                double_batch_trick=FLAGS.double_batch_trick,
                strategy=strategy)
          elif FLAGS.pretrain_loss == 'contrastive_ceb':
            con_loss, logits_con, labels_con, output_metrics = obj_lib.add_contrastive_ceb_loss(
                proj_outputs,
                kappa_e=FLAGS.kappa_e,
                kappa_b=FLAGS.kappa_b,
                beta=ceb_beta,
                sampling=FLAGS.ceb_sampling,
                double_batch_trick=FLAGS.double_batch_trick,
                strategy=strategy)
          elif FLAGS.pretrain_loss == 'byol':
            con_loss, output_metrics = obj_lib.add_byol_loss(
                proj_outputs, byol_loss_weight=FLAGS.byol_loss_weight)
            con_loss2, _ = obj_lib.add_byol_loss(
                proj_outputs2, byol_loss_weight=FLAGS.byol_loss_weight)
            con_loss += con_loss2
          elif FLAGS.pretrain_loss == 'byol_ceb':
            con_loss, output_metrics = obj_lib.add_byol_ceb_loss(
                proj_outputs,
                proj_outputs2,
                target_proj_output,
                target_proj_output2,
                z_mu,
                z_mu2,
                kappa_e=FLAGS.kappa_e,
                kappa_b=FLAGS.kappa_b,
                beta=ceb_beta,
                byol_loss_weight=FLAGS.byol_loss_weight)
          else:
            raise NotImplementedError
          loss = con_loss if loss is None else loss + con_loss

          ceb_beta_metric.update_state(ceb_beta)
          info_metrics = {'i_xzx_y': i_xzx_y_metric,
                          'i_y_zx': i_y_zx_metric,
                          'i_yzy_x': i_yzy_x_metric,
                          'i_x_zy': i_x_zy_metric,
                          'h_e_zx_x': h_e_zx_x_metric,
                          'h_b_zx_y': h_b_zx_y_metric,
                          'byol_loss': byol_loss_metric}
          for k, m in info_metrics.items():
            if k in output_metrics:
              m.update_state(tf.stop_gradient(output_metrics[k]))

          if FLAGS.pretrain_loss in ['byol', 'byol_ceb']:
            contrast_loss_metric.update_state(tf.stop_gradient(con_loss))
          else:
            metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                  contrast_acc_metric,
                                                  contrast_entropy_metric,
                                                  tf.stop_gradient(con_loss),
                                                  tf.stop_gradient(logits_con),
                                                  tf.stop_gradient(labels_con))

        if supervised_head_outputs is not None:
          l = labels['labels']
          if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            if not FLAGS.use_momentum_encoder:
              l = tf.concat([l, l], 0)
          sup_loss = obj_lib.add_supervised_loss(
              labels=l, logits=supervised_head_outputs)
          loss = sup_loss if loss is None else loss + sup_loss

          metrics.update_finetune_metrics_train(
              supervised_loss_metric,
              supervised_acc_metric,
              tf.stop_gradient(sup_loss),
              tf.stop_gradient(l),
              tf.stop_gradient(supervised_head_outputs))
        weight_decay = model_util.add_weight_decay(
            model, adjust_per_optimizer=True)
        weight_decay_metric.update_state(weight_decay)
        loss += weight_decay
        total_loss_metric.update_state(tf.stop_gradient(loss))
        # The default behavior of `apply_gradients` is to sum gradients from all
        # replicas so we divide the loss by the number of replicas so that the
        # mean gradient is applied.
        loss = loss / strategy.num_replicas_in_sync
        if FLAGS.explicitly_watch_vars:
          grads = tape.gradient(loss, trainable_variables)  # BYOL
          optimizer.apply_gradients(zip(grads, trainable_variables))  # BYOL
        else:
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if FLAGS.use_momentum_encoder and FLAGS.train_mode == 'pretrain':
        # global_step = optimizer.iterations
        ema_alpha_value = ema_alpha_fn(tf.cast(global_step, dtype=tf.float32))
        ema_alpha_metric.update_state(ema_alpha_value)
        model.update_ema(ema_alpha_value)

    with strategy.scope():
      steps_per_loop = checkpoint_steps

      @tf.function
      def train_multiple_steps(iterator):
        # `tf.range` is needed so that this runs in a `tf.while_loop` and is
        # not unrolled.
        for _ in tf.range(steps_per_loop):
          # Drop the "while" prefix created by tf.while_loop which otherwise
          # gets prefixed to every variable name. This does not affect training
          # but does affect the checkpoint conversion script.
          # TODO(b/161712658): Remove this.
          with tf.name_scope(''):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            strategy.run(single_step, (features, labels))

      global_step = optimizer.iterations
      cur_step = global_step.numpy()
      iterator = iter(ds)
      while cur_step < train_steps:
        train_multiple_steps(iterator)
        cur_step = global_step.numpy()
        checkpoint_manager.save(cur_step)
        logging.info('Completed: %d / %d steps', cur_step, train_steps)
        metrics.log_and_write_metrics(all_metrics, cur_step, summary_writer)
        if FLAGS.learning_rate_schedule != 'constant':
          summary_writer.scalar(
              'learning_rate',
              learning_rate(tf.cast(global_step, dtype=tf.float32)),
              global_step)
        if FLAGS.use_momentum_encoder:
          summary_writer.scalar(
              'train/ema_alpha',
              ema_alpha_fn(tf.cast(global_step, dtype=tf.float32)),
              global_step)
        summary_writer.flush()
        for metric in all_metrics:
          metric.reset_states()
      logging.info('Training complete...')
      pool.close()
      pool.join()

      if FLAGS.export_train_model:
        save(model, global_step=cur_step, suffix='train')

    if FLAGS.mode == 'train_then_eval':
      perform_evaluation(model, builder, eval_steps,
                         checkpoint_manager.latest_checkpoint, strategy,
                         topology)

if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
