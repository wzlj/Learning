# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

models_path = os.path.join("_path/models/")
sys.path.append(models_path)

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_model
from official.resnet import resnet_run_loop


HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 16
NUM_CLASSES = 5


# TODO(tobyboyd): Change to best practice 45K(train)/5K(val)/10K(test) splits.
NUM_IMAGES = {
    'train': 2500,
    'validation': 1500,
}

DATASET_NAME = 'Five Class'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [os.path.join(data_dir, 'train224.tfrecords')]
  else:
    return [os.path.join(data_dir, 'val224.tfrecords')]


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(
        image, HEIGHT + 16, WIDTH + 16)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  # image = tf.image.per_image_standardization(image)

  return image/255.0


def parse_record(raw_record, is_training, dtype):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.io.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [NUM_CHANNELS, HEIGHT, WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)
  image = tf.cast(image, dtype)

  return image, label


def _parse_record(example_proto, is_training, dtype):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.io.decode_raw(features['image_raw'], tf.uint8)
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    label = tf.cast(features['label'], tf.int32)


    tf_image = tf.reshape(tf_image, [HEIGHT, WIDTH, NUM_CHANNELS])

    image = tf.cast(tf_image, tf.float32)
    image = preprocess_image(image, is_training)
    image = tf.cast(image, dtype)

    # label = tf.cast(features['label'], tf.int32)
    return image, label


def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=_parse_record,
             input_context=None,
             drop_remainder=False):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.TFRecordDataset(filenames)
  # dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  # if input_context:
  #   tf.compat.v1.logging.info(
  #       'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
  #           input_context.input_pipeline_id, input_context.num_input_pipelines))
  #   dataset = dataset.shard(input_context.num_input_pipelines,
  #                           input_context.input_pipeline_id)

  # if is_training:
  #   # Shuffle the input files
  #   dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=NUM_IMAGES['train'],
      parse_record_fn=parse_record_fn,
      num_epochs=num_epochs,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      drop_remainder=drop_remainder
  )


###############################################################################
# Running the model
###############################################################################
class Pint_Resnet(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(Pint_Resnet, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  if params['fine_tune']:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'] * params.get('num_workers', 1),
      batch_denom=256, num_images=NUM_IMAGES['train'],
      boundary_epochs=[30, 60, 80, 90], decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
      warmup=warmup, base_lr=base_lr)

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Pint_Resnet,
      resnet_size=params['resnet_size'],
      weight_decay=flags.FLAGS.weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
      label_smoothing=flags.FLAGS.label_smoothing
  )


def define_mydata_flags():
  resnet_run_loop.define_resnet_flags(
      resnet_size_choices=['18', '34', '50', '101', '152', '200'],
      dynamic_loss_scale=True,
      fp16_implementation=True)
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(train_epochs=20,
                          use_train_and_evaluate=False,
                          data_format='channels_last',
                          )
  flags_core.set_defaults(data_dir='_path/record/')


def run_resnet(flags_obj):
  """Run ResNet CIFAR-10 training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.

  Returns:
    Dictionary of results. Including final accuracy.
  """
  if flags_obj.image_bytes_as_serving_input:
    tf.compat.v1.logging.fatal(
        '--image_bytes_as_serving_input cannot be set to True for CIFAR. '
        'This flag is only applicable to ImageNet.')
    return

  result = resnet_run_loop.resnet_main(
      flags_obj, resnet_model_fn, input_fn, DATASET_NAME,
      shape=[HEIGHT, WIDTH, NUM_CHANNELS])

  return result


def main(_):
  with logger.benchmark_context(flags.FLAGS):
      run_resnet(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_mydata_flags()   # define the parameters
  absl_app.run(main)





