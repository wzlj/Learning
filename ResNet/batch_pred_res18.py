# coding=utf-8

import tensorflow as tf
import numpy as np
import pdb
import cv2
import os
import time
import glob
import ResNet_

# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# import tensorflow.contrib.slim as slim
import SqueezeNet
from create_tf_record import *

flags = tf.app.flags

flags.DEFINE_string('images_dir',
                    './test_image',
                    'Path to images (directory).')
flags.DEFINE_string('models',
                    './models/best_models.ckpt',
                    'Path to model (directory).')

flags.DEFINE_string('label',
                    './label.txt',
                    'Path to label (directory).')
FLAGS = flags.FLAGS


def preprocess_image(image, output_height, output_width,
                        add_image_summaries=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    add_image_summaries: Enable image summaries.

  Returns:
    A preprocessed image.
  """
  if add_image_summaries:
    tf.summary.image('image', tf.expand_dims(image, 0))
  # Transform the image to floats.
  image = tf.to_float(image)

  # Resize and crop if needed.
  resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                         output_width,
                                                         output_height)
  if add_image_summaries:
    tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

  # Subtract off the mean and divide by the variance of the pixels.
  return tf.image.per_image_standardization(resized_image)

def batch_pred(models_path, images_list, labels_nums, data_format):

    [batch_size, resize_height, resize_width, depths] = data_format
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    out = ResNet_.resnet_v1_18(inputs=input_images, num_classes=labels_nums, is_training=False)

    score = tf.nn.softmax(out, name='pre')
    class_id = tf.argmax(score, 1)

    gpu_options = tf.GPUOptions(allow_growth=False)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, models_path)
        tot = len(images_list)

        for idx in range(0, tot, batch_size):
            images = list()
            idx_end = min(tot, idx + batch_size)
            print(idx)
            for i in range(idx, idx_end):
                image_path = images_list[i]
                image = open(image_path, 'rb').read()
                image = tf.image.decode_jpeg(image, channels=3)
                processed_image = preprocess_image(image, resize_height, resize_width)
                processed_image = sess.run(processed_image)
                # print("processed_image.shape", processed_image.shape)
                images.append(processed_image)
            images = np.array(images)
            start = time.time()
            sess.run([score, class_id], feed_dict={input_images: images})
            end = time.time()
            print("time of batch {} is %f".format(batch_size) % (end - start))

    sess.close()


def get_images(image_dir):
    imgs_list = []

    for filename in os.listdir(image_dir):
        if filename.split(".")[-1] == "jpg":
            imgs_list.append(os.path.join(image_dir, filename))

    return imgs_list


def main(_):

    if False:
        image_dir = FLAGS.images_dir
        labels_filename = FLAGS.label
        models_path = FLAGS.models
    else:
        image_dir = '/home/jlai/study/tensorflow_models_learning/test_imgs/'
        labels_filename = 'dataset/label.txt'
        models_path = '/home/swshare/model/ResNet18_v1/best_models.ckpt'

    images_list = get_images(image_dir)

    labels_nums = 5
    batch_size = 1  #
    resize_height = 224  # the height of input image
    resize_width = 224  # the width of input image
    depths = 3
    data_format = [batch_size, resize_height, resize_width, depths]
    # predict(models_path, image_dir, labels_filename, class_nums, data_format)
    batch_pred(models_path, images_list, labels_nums, data_format)


if __name__ == '__main__':
    tf.app.run()





