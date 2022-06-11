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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags

import data_util as data_util
import tensorflow.compat.v1 as tf
import math

FLAGS = flags.FLAGS


def pad_to_batch(dataset, batch_size):
  """Pad Tensors to specified batch size.

  Args:
    dataset: An instance of tf.data.Dataset.
    batch_size: The number of samples per batch of input requested.

  Returns:
    An instance of tf.data.Dataset that yields the same Tensors with the same
    structure as the original padded to batch_size along the leading
    dimension.

  Raises:
    ValueError: If the dataset does not comprise any tensors; if a tensor
      yielded by the dataset has an unknown number of dimensions or is a
      scalar; or if it can be statically determined that tensors comprising
      a single dataset element will have different leading dimensions.
  """
  def _pad_to_batch(*args):
    """Given Tensors yielded by a Dataset, pads all to the batch size."""
    flat_args = tf.nest.flatten(args)

    for tensor in flat_args:
      if tensor.shape.ndims is None:
        raise ValueError(
            'Unknown number of dimensions for tensor %s.' % tensor.name)
      if tensor.shape.ndims == 0:
        raise ValueError('Tensor %s is a scalar.' % tensor.name)

    # This will throw if flat_args is empty. However, as of this writing,
    # tf.data.Dataset.map will throw first with an internal error, so we do
    # not check this case explicitly.
    first_tensor = flat_args[0]
    first_tensor_shape = tf.shape(first_tensor)
    first_tensor_batch_size = first_tensor_shape[0]
    difference = batch_size - first_tensor_batch_size

    for i, tensor in enumerate(flat_args):
      control_deps = []
      if i != 0:
        # Check that leading dimensions of this tensor matches the first,
        # either statically or dynamically. (If the first dimensions of both
        # tensors are statically known, the we have to check the static
        # shapes at graph construction time or else we will never get to the
        # dynamic assertion.)
        if (first_tensor.shape[:1].is_fully_defined() and
            tensor.shape[:1].is_fully_defined()):
          if first_tensor.shape[0] != tensor.shape[0]:
            raise ValueError(
                'Batch size of dataset tensors does not match. %s '
                'has shape %s, but %s has shape %s' % (
                    first_tensor.name, first_tensor.shape,
                    tensor.name, tensor.shape))
        else:
          curr_shape = tf.shape(tensor)
          control_deps = [tf.Assert(
              tf.equal(curr_shape[0], first_tensor_batch_size),
              ['Batch size of dataset tensors %s and %s do not match. '
               'Shapes are' % (tensor.name, first_tensor.name), curr_shape,
               first_tensor_shape])]

      with tf.control_dependencies(control_deps):
        # Pad to batch_size along leading dimension.
        flat_args[i] = tf.pad(
            tensor, [[0, difference]] + [[0, 0]] * (tensor.shape.ndims - 1))
      flat_args[i].set_shape([batch_size] + tensor.shape.as_list()[1:])

    return tf.nest.pack_sequence_as(args, flat_args)

  return dataset.map(_pad_to_batch)


def build_input_fn(builder, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """
  def _input_fn(params):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if FLAGS.train_mode == 'pretrain':
        if FLAGS.augmentation_mode == 'augmentation_based':
            # TODO: somehow un-hardcode '4'
            num_augs = 4
            aug_num = tf.random.uniform(shape=[], minval=0, maxval=num_augs, dtype=tf.dtypes.int32)
            image = preprocess_fn_pretrain(
                image,
                augmentation_mode='augmentation_based',
                augmentation_num=aug_num
            )

            label = tf.one_hot(aug_num, num_augs)
            return image, label, 1.0
        elif FLAGS.augmentation_mode == 'augmentation_based2':
            # TODO: somehow un-hardcode '4'
            num_augs = 4
            aug_num1 = tf.random.uniform(shape=[], minval=0, maxval=num_augs, dtype=tf.dtypes.int32)
            aug_num2 = tf.random.uniform(shape=[], minval=0, maxval=num_augs, dtype=tf.dtypes.int32)
            image = preprocess_fn_pretrain(
                image,
                augmentation_mode='augmentation_based2',
                augmentation_num=(aug_num1, aug_num2)
            )

            label = (tf.one_hot(aug_num1, num_augs) + tf.one_hot(aug_num2, num_augs))
            label = tf.math.l2_normalize(label, -1)

            return image, label, 1.0
        elif FLAGS.augmentation_mode == 'augmentation_diff':
          label = tf.zeros([num_classes])
          return image, label, 1.0
        else:
          xs = []
          for _ in range(2):  # Two transformations
            xs.append(preprocess_fn_pretrain(image))
          image = tf.concat(xs, -1)
          label = tf.zeros([num_classes])
      else:
        image = preprocess_fn_finetune(image)
        label = tf.one_hot(label, num_classes)

      return image, label, 1.0

    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training, as_supervised=True)
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(map_fn,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if FLAGS.augmentation_mode.startswith('augmentation_based'):
      batch_size = params['batch_size'] * 2
    else:
      batch_size = params['batch_size']
    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    if FLAGS.augmentation_mode.startswith('augmentation_diff'):
        def group_pairs_fn(images, labels, weights):
            # group images into pairs, each pair will have same augmentation applied
            images = tf.reshape(images, (images.shape[0] // 2, 2, *images.shape[1:]))
            labels = tf.reshape(labels, (labels.shape[0] // 2, 2, *labels.shape[1:]))
            weights = tf.reshape(weights, (weights.shape[0] // 2, 2, *weights.shape[1:]))
            return images, labels, weights

        def augment_pairs_fn(images, labels, weights):
            # apply exactly same augmentation on a pair of images
            # and return both images with and without augmentations

            # TODO: somehow un-hardcode '4'
            num_augs = 4
            aug_num1 = tf.random.uniform(shape=[], minval=0, maxval=num_augs, dtype=tf.dtypes.int32)
            aug_num2 = tf.random.uniform(shape=[], minval=0, maxval=num_augs, dtype=tf.dtypes.int32)

            # apply augmentations on a single image with C*2 channels
            # so that both images in pair have exactly same augmentations applied
            image_pair_as_one = tf.transpose(images, (1, 2, 0, 3))
            image_pair_as_one = tf.reshape(image_pair_as_one, (*image_pair_as_one.shape[:2], -1))

            augmented_images = preprocess_fn_pretrain(
                image_pair_as_one,
                augmentation_mode='augmentation_based2',
                augmentation_num=(aug_num1, aug_num2)
            )

            augmented_images = tf.reshape(augmented_images, (*augmented_images.shape[:2], 2, 3))
            augmented_images = tf.transpose(augmented_images, (2, 0, 1, 3))

            # augmented_images = tf.stack(augmented_images, axis=0)
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
            images = tf.concat([images, augmented_images], axis=0)
            labels = tf.repeat(labels, 2, axis=0)
            weights = tf.repeat(weights, 2, axis=0)

            return images, labels, weights

        def regroup_pairs_fn(images, labels, weights):
            # join pairs of images with same augmentations (2 images, augmented and non-augmented = 4 samples)
            # so that they can be inputted into the model
            images = tf.reshape(images, (images.shape[0] * 4, *images.shape[2:]))
            labels = tf.reshape(labels, (labels.shape[0] * 4, *labels.shape[2:]))
            weights = tf.reshape(weights, (weights.shape[0] * 4, *weights.shape[2:]))
            return images, labels, weights

        dataset = dataset.map(group_pairs_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()
        dataset = dataset.map(augment_pairs_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size // 2, drop_remainder=is_training)
        dataset = dataset.map(regroup_pairs_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = pad_to_batch(dataset, batch_size * 2)
    else:
        dataset = pad_to_batch(dataset, batch_size)

    images, labels, mask = tf.data.make_one_shot_iterator(dataset).get_next()

    return images, {'labels': labels, 'mask': mask}
  return _input_fn


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
