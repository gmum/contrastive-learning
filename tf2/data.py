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
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import data_util
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def build_input_fn(builder, global_batch_size, topology, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if FLAGS.train_mode == 'pretrain' and FLAGS.augmentation_mode.startswith('augmentation_diff'):
        label = tf.zeros([num_classes])
        return image, label
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label

    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if FLAGS.augmentation_mode.startswith('augmentation_diff') and FLAGS.train_mode == 'pretrain':
        dataset = dataset.batch(2, drop_remainder=is_training)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=is_training)

    if FLAGS.augmentation_mode.startswith('augmentation_diff') and FLAGS.train_mode == 'pretrain':
        def group_pairs_fn(images, labels):
            # group images into pairs, each pair will have same augmentation applied
            # (2, 32, 32, 3)
            images = tf.reshape(images, (images.shape[0] // 2, 2, *images.shape[1:]))
            labels = tf.reshape(labels, (labels.shape[0] // 2, 2, *labels.shape[1:]))
            # (1, 2, 32, 32, 3)

            # replicate image pairs to fill batch size
            images = tf.repeat(images, batch_size // 2, 0)
            labels = tf.repeat(labels, batch_size // 2, 0)
            # (batch_size // 2, 2, 32, 32, 3)
            return images, labels

        def augment_pairs_fn(images, labels):
            # apply exactly same augmentations on a pair of images
            # and return both images with and without augmentations

            # apply augmentations on a single image with C*2 channels
            # so that both images in pair have exactly same augmentations applied
            # (2, 32, 32, 3)
            image_pair_as_one = tf.transpose(images, (1, 2, 0, 3)) # (32, 32, 2, 3)
            image_pair_as_one = tf.reshape(image_pair_as_one, (*image_pair_as_one.shape[:2], -1))  # (32, 32, 6)

            augmented_image_pairs = []
            for _ in range(2):
                augmented_images = preprocess_fn_pretrain(image_pair_as_one)  # (32, 32, 6)
                augmented_images = tf.reshape(augmented_images, (*augmented_images.shape[:2], 2, 3))  # (32, 32, 2, 3)
                augmented_images = tf.transpose(augmented_images, (2, 0, 1, 3))  # (2, 32, 32, 3)
                augmented_image_pairs.append(augmented_images)

            images = tf.concat(augmented_image_pairs, axis=0)  # (4, 32, 32, 3)
            labels = tf.repeat(labels, 2, axis=0) # (4, )

            return images, labels

        def regroup_pairs_fn(images, labels):
            # join pairs of images with same augmentations (2 images, augmented and non-augmented = 4 samples)
            # so that they can be inputted into the model
            images = tf.reshape(images, (images.shape[0] * 4, *images.shape[2:]))
            labels = tf.reshape(labels, (labels.shape[0] * 4, *labels.shape[2:]))
            return images, labels

        dataset = dataset.map(group_pairs_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()
        dataset = dataset.map(augment_pairs_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size // 2, drop_remainder=is_training)
        dataset = dataset.map(regroup_pairs_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)
