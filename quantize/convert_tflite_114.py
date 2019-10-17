import os
import numpy as np
import tensorflow as tf
import resnet_main
# if commented tf.compat.v1.enable_eager_execution(), run this file and resulted this error:
'''
lib/python3.7/site-packages/tensorflow/lite/python/optimize/tensorflow_lite_wrap_calibration_wrapper.py", line 112, in FeedTensor
    return _tensorflow_lite_wrap_calibration_wrapper.CalibrationWrapper_FeedTensor(self, input_value)
ValueError: Cannot set tensor: Got tensor of type STRING but expected type FLOAT32 for input 2, name: input_tensor
''' 
tf.compat.v1.enable_eager_execution()  


HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 16
NUM_CLASSES = 5
data_dir = '/home/jlai/data/five_class/record/'

# TODO(tobyboyd): Change to best practice 45K(train)/5K(val)/10K(test) splits.
NUM_IMAGES = {
    'train': 2500,
    'validation': 1500,
}

# POST TRAINING QUANTIZATION
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
  image = tf.image.per_image_standardization(image)
  return image

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
    image = tf.cast(image, tf.float32)

    # label = tf.cast(features['label'], tf.int32)
    return image



if __name__ == "__main__":

    saved_model_dir = "/home/jlai/study/models/official/resnet/model/1571281017"  # "./savedmodel/1564619423"
    converted_dir = './converted_model'
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)

    is_training = False
    batch_size = 1
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda value: _parse_record(value, is_training, 'float32'))
    dataset = dataset.batch(batch_size=1)
    def representative_dataset_gen():
        for i, input_value in enumerate(dataset):
            # input_value = input_value
            # print("___________________")
            # # print(input_value)
            # print("___________________")
            print("___________________{}th ___________________".format(i))
            yield [input_value]
        # for i in range(10):
        #     value = np.random.normal(0, 1, size=(1, 224,224,3))
        #     value = value.astype('float32')
        #     print(value)
        #     yield [ value ]
        pass

    converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print("+++")
    print(output_details)
    open(converted_dir + '/' + "pint_resnet18.tflite", "wb").write(tflite_model)
