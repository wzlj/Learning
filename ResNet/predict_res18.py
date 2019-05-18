#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob

# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# import tensorflow.contrib.slim as slim
import ResNet_

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


def read_image(bgr_image, resize_height, resize_width, normalization=False):
    '''
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization: normalization or not
    :return: the data of image
    '''

    # bgr_image = cv2.imread(filename)
    if len(bgr_image.shape) == 2:  # if the image is gray, convert it to bgr format
        print("Warning:gray image", bgr_image)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        rgb_image = rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image

def predict(models_path, image_dir, labels_filename, labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')

    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    # model
    # with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    out = ResNet_.resnet_v1_18(inputs=input_images, num_classes=labels_nums, is_training=False)

    # out = tf.squeeze(out, [1, 2])
    score = tf.nn.softmax(out, name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for image_path in images_list:
        img = cv2.imread(image_path)
        
        im = read_image(img, resize_height, resize_width, normalization=True)
        im = im[np.newaxis, :]
        pre_score, pre_label = sess.run([score, class_id], feed_dict={input_images: im})
        max_score = pre_score[0, pre_label]
        print("{} is: pre labels:{}, score: {}".format(image_path, pre_label, max_score))
        print(str(labels[pre_label]))
        res = '{:s} {:.3f}'.format(str(labels[pre_label]), float(max_score))
        cv2.putText(img, res, (int(img.shape[1]/15), 25), font, 0.8, (0, 0, 255), 2)
        cv2.imshow("demo", img)
        cv2.waitKey(0)
    sess.close()


def main(_):
    image_dir = FLAGS.images_dir
    labels_filename = FLAGS.label
    models_path = FLAGS.models

    class_nums = 5
    batch_size = 1  #
    resize_height = 224  # the height of input image
    resize_width = 224  # the width of input image
    depths = 3
    data_format = [batch_size, resize_height, resize_width, depths]
    predict(models_path, image_dir, labels_filename, class_nums, data_format)


if __name__ == '__main__':
    tf.app.run()
