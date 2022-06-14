# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contrastive loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import
from absl import flags

from model_util import get_projection_head

LARGE_NUM = 1e9
FLAGS = flags.FLAGS


def add_supervised_loss(labels, logits, weights, **kwargs):
  """Compute loss for model and add it to loss collection."""
  return tf.losses.softmax_cross_entropy(labels, logits, weights, **kwargs)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0,
                         labels=None):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    if FLAGS.augmentation_mode == 'augmentation_diff_combined':
      hidden = [tf.math.l2_normalize(hidden[0], -1), tf.math.l2_normalize(hidden[1], -1)]
    else:
      hidden = tf.math.l2_normalize(hidden, -1)

  if FLAGS.augmentation_mode == 'augmentation_diff':
      hidden = tf.reshape(hidden, (hidden.shape[0] // 4, 4, *hidden.shape[1:]))
      batch_size = hidden.shape[0]

      # element 0, 1 = augmentation1. 2, 3 = augmentation2
      hidden1 = tf.math.l2_normalize(hidden[:, 2] - hidden[:, 0], -1)
      hidden2 = tf.math.l2_normalize(hidden[:, 3] - hidden[:, 1], -1)

      labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
      masks = tf.one_hot(tf.range(batch_size), batch_size)

      hidden1_large = hidden1
      hidden2_large = hidden2
  elif FLAGS.augmentation_mode == 'augmentation_diff_combined':
      hidden_img, hidden_aug = hidden[0], hidden[1]

      hidden_img = tf.reshape(hidden_img, (hidden_img.shape[0] // 4, 4, *hidden_img.shape[1:]))
      hidden_aug = tf.reshape(hidden_aug, (hidden_aug.shape[0] // 4, 4, *hidden_aug.shape[1:]))
      batch_size = hidden_aug.shape[0]

      # element 0, 1 = augmentation1. 2, 3 = augmentation2
      # element 0, 2 = image1. 1, 3 = image2
      hidden_aug_left = tf.concat([hidden_aug[:, 0], hidden_aug[:, 1]], 0)
      hidden_aug_right = tf.concat([hidden_aug[:, 2], hidden_aug[:, 3]], 0)

      aug_diff_proj_head = get_projection_head(hidden_aug_left - hidden_aug_right, is_training=True,
                                               mid_dim=hidden_aug.shape[-1])
      aug_diff_proj_head = tf.math.l2_normalize(aug_diff_proj_head, -1)
      hidden_aug1 = aug_diff_proj_head[:aug_diff_proj_head.shape[0]//2]
      hidden_aug2 = aug_diff_proj_head[aug_diff_proj_head.shape[0]//2:]

      hidden_img1 = tf.concat([hidden_img[:, 0], hidden_img[:, 1]], 0)
      hidden_img2 = tf.concat([hidden_img[:, 2], hidden_img[:, 3]], 0)

      labels_img = tf.one_hot(tf.range(batch_size * 2), batch_size * 4)
      masks_img = tf.one_hot(tf.range(batch_size * 2), batch_size * 2)

      logits_aa_img = tf.matmul(hidden_img1, hidden_img1, transpose_b=True) / temperature
      logits_aa_img = logits_aa_img - masks_img * LARGE_NUM
      logits_bb_img = tf.matmul(hidden_img2, hidden_img2, transpose_b=True) / temperature
      logits_bb_img = logits_bb_img - masks_img * LARGE_NUM
      logits_ab_img = tf.matmul(hidden_img1, hidden_img2, transpose_b=True) / temperature
      logits_ba_img = tf.matmul(hidden_img2, hidden_img1, transpose_b=True) / temperature

      loss_a_img = tf.losses.softmax_cross_entropy(
          labels_img, tf.concat([logits_ab_img, logits_aa_img], 1), weights=weights)
      loss_b_img = tf.losses.softmax_cross_entropy(
          labels_img, tf.concat([logits_ba_img, logits_bb_img], 1), weights=weights)
      loss_img = loss_a_img + loss_b_img

      labels_aug = tf.one_hot(tf.range(batch_size), batch_size * 2)
      masks_aug = tf.one_hot(tf.range(batch_size), batch_size)

      logits_aa_aug = tf.matmul(hidden_aug1, hidden_aug1, transpose_b=True) / temperature
      logits_aa_aug = logits_aa_aug - masks_aug * LARGE_NUM
      logits_bb_aug = tf.matmul(hidden_aug2, hidden_aug2, transpose_b=True) / temperature
      logits_bb_aug = logits_bb_aug - masks_aug * LARGE_NUM
      logits_ab_aug = tf.matmul(hidden_aug1, hidden_aug2, transpose_b=True) / temperature
      logits_ba_aug = tf.matmul(hidden_aug2, hidden_aug1, transpose_b=True) / temperature

      loss_a_aug = tf.losses.softmax_cross_entropy(
          labels_aug, tf.concat([logits_ab_aug, logits_aa_aug], 1), weights=weights)
      loss_b_aug = tf.losses.softmax_cross_entropy(
          labels_aug, tf.concat([logits_ba_aug, logits_bb_aug], 1), weights=weights)
      loss_aug = loss_a_aug + loss_b_aug

      loss = FLAGS.pretrain_loss_weight_aug * loss_aug + (1.0 - FLAGS.pretrain_loss_weight_aug) * loss_img

      logits_ab_aug = tf.concat([logits_ab_aug, tf.fill(logits_ab_aug.shape, -LARGE_NUM)], -1)
      labels_aug = tf.concat([labels_aug, tf.zeros_like(labels_aug)], -1)

      logits_ab = tf.concat([logits_ab_img, logits_ab_aug], axis=0)
      labels = tf.concat([labels_img, labels_aug], axis=0)

      return loss, logits_ab, labels
  else:
      hidden1, hidden2 = tf.split(hidden, 2, 0)
      batch_size = tf.shape(hidden1)[0]

      # Gather hidden1/hidden2 across replicas and create local labels.
      if tpu_context is not None:
        hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
        hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
        enlarged_batch_size = tf.shape(hidden1_large)[0]
        # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
        replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
        masks = tf.one_hot(labels_idx, enlarged_batch_size)
      else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
  loss_b = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
  loss = loss_a + loss_b

  return loss, logits_ab, labels


def tpu_cross_replica_concat(tensor, tpu_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    tpu_context: A `TPUContext`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if tpu_context is None or tpu_context.num_replicas <= 1:
    return tensor

  num_replicas = tpu_context.num_replicas

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
