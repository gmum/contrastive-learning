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

from absl import flags

import tensorflow.compat.v2 as tf

from model import ProjectionHead

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits):
    """Compute mean supervised loss over local batch."""
    losses = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                    logits)
    return tf.reduce_mean(losses)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         strategy=None,
                         projection_head_diff=None):
    """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    strategy: context information for tpu.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    if FLAGS.augmentation_mode == 'augmentation_diff':
        hidden = tf.reshape(hidden, (hidden.shape[0] // 4, 4, *hidden.shape[1:]))
        batch_size = hidden.shape[0]

        # element 0, 1 = augmentation1. 2, 3 = augmentation2
        diff1 = hidden[:, 2] - hidden[:, 0]
        diff2 = hidden[:, 3] - hidden[:, 1]

        diff = tf.concat([diff1, diff2], 0)

        diff, _ = projection_head_diff(
            diff, training=True)
        diff = tf.math.l2_normalize(diff, -1)

        diff1 = diff[:diff.shape[0] // 2]
        diff2 = diff[diff.shape[0] // 2:]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        hidden1 = diff1
        hidden2 = diff2
        hidden1_large = diff1
        hidden2_large = diff2
    elif FLAGS.augmentation_mode == 'augmentation_diff_combined':
        hidden = tf.reshape(hidden, (hidden.shape[0] // 4, 4, *hidden.shape[1:]))
        batch_size = hidden.shape[0]

        if FLAGS.double_head:
            projection_head_aug = ProjectionHead()
            hidden_aug, _ = projection_head_aug(hidden, training=True)
        else:
            hidden_aug = hidden

        # element 0, 1 = augmentation1. 2, 3 = augmentation2
        # element 0, 2 = image1. 1, 3 = image2
        hidden_aug_left = tf.concat([hidden_aug[:, 0], hidden_aug[:, 1]], 0)
        hidden_aug_right = tf.concat([hidden_aug[:, 2], hidden_aug[:, 3]], 0)

        diff = hidden_aug_left - hidden_aug_right

        aug_diff_proj_head, _ = projection_head_diff(diff, training=True)
        aug_diff_proj_head = tf.math.l2_normalize(aug_diff_proj_head, -1)
        hidden_aug1 = aug_diff_proj_head[:aug_diff_proj_head.shape[0] // 2]
        hidden_aug2 = aug_diff_proj_head[aug_diff_proj_head.shape[0] // 2:]

        if FLAGS.double_head:
            projection_head_img = ProjectionHead()
            hidden_img, _ = projection_head_img(hidden, training=True)
        else:
            hidden_img = hidden

        hidden_img1 = tf.concat([hidden_img[:, 0], hidden_img[:, 1]], 0)
        hidden_img2 = tf.concat([hidden_img[:, 2], hidden_img[:, 3]], 0)

        labels_img = tf.one_hot(tf.range(batch_size * 2), batch_size * 2)

        logits_ab_img = tf.matmul(hidden_img1, hidden_img2, transpose_b=True) / temperature
        logits_ba_img = tf.matmul(hidden_img2, hidden_img1, transpose_b=True) / temperature

        loss_a_img = tf.nn.softmax_cross_entropy_with_logits(
            labels_img, logits_ab_img)
        loss_b_img = tf.nn.softmax_cross_entropy_with_logits(
            labels_img, logits_ba_img)
        loss_img = loss_a_img + loss_b_img
        loss_img = tf.reduce_mean(loss_img)

        labels_aug = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks_aug = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa_aug = tf.matmul(hidden_aug1, hidden_aug1, transpose_b=True) / temperature
        logits_aa_aug = logits_aa_aug - masks_aug * LARGE_NUM
        logits_bb_aug = tf.matmul(hidden_aug2, hidden_aug2, transpose_b=True) / temperature
        logits_bb_aug = logits_bb_aug - masks_aug * LARGE_NUM
        logits_ab_aug = tf.matmul(hidden_aug1, hidden_aug2, transpose_b=True) / temperature
        logits_ba_aug = tf.matmul(hidden_aug2, hidden_aug1, transpose_b=True) / temperature

        loss_a_aug = tf.nn.softmax_cross_entropy_with_logits(
            labels_aug, tf.concat([logits_ab_aug, logits_aa_aug], 1))
        loss_b_aug = tf.nn.softmax_cross_entropy_with_logits(
            labels_aug, tf.concat([logits_ba_aug, logits_bb_aug], 1))
        loss_aug = loss_a_aug + loss_b_aug
        loss_aug = tf.reduce_mean(loss_aug)

        loss = FLAGS.pretrain_loss_weight_aug * loss_aug + (1.0 - FLAGS.pretrain_loss_weight_aug) * loss_img

        logits_ab_aug = tf.concat([logits_ab_aug, tf.fill(logits_ab_aug.shape, -LARGE_NUM)], -1)
        # labels_aug = tf.concat([labels_aug, tf.zeros_like(labels_aug)], -1)

        logits_ab = tf.concat([logits_ab_img, logits_ab_aug], axis=0)
        labels = tf.concat([labels_img, labels_aug], axis=0)

        return loss, logits_ab, labels
    else:
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if strategy is not None:
            hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
            hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
            enlarged_batch_size = tf.shape(hidden1_large)[0]
            # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
            replica_context = tf.distribute.get_replica_context()
            replica_id = tf.cast(
                tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
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

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)

    return loss, logits_ab, labels


def tpu_cross_replica_concat(tensor, strategy=None):
    """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.

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

        # Flatten the replica dimension.
        # The first dimension size will be: tensor.shape[0] * num_replicas
        # Using [-1] trick to support also scalar input.
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
