#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import slim.nets.resnet_v1 as resnet_v1
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1

from create_tf_record import *
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1

def predict(models_path, image_dir, labels_filename, labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    # model
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        out, end_points = resnet_v1.resnet_v1_50(inputs=input_images, num_classes=labels_nums, is_training=False)

    # out = tf.squeeze(out, )
    # out = tf.squeeze(out, [1, 2])
    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out, name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list = glob.glob(os.path.join(image_dir, '*.jpg'))


    for image_path in images_list:
        im = read_image(image_path, resize_height, resize_width, normalization=True)
        im = im[np.newaxis, :]

        pre_score, out_,  pre_label = sess.run([score, out,  class_id], feed_dict={input_images: im})
        print("out_  =============> \n", out_)

        # pre_score, pre_label = sess.run([score, class_id], feed_dict={input_images: im})

        max_score = pre_score[0, pre_label]

        print("_______________________________________________________________________\n\n")
        print("{} is: pre labels:{}, score: {}".format(image_path, pre_label,   max_score))
        # print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_label, labels[pre_label], max_score))
    sess.close()


if __name__ == '__main__':

    class_nums = 5
    image_dir = 'test_image'
    labels_filename = 'dataset/label.txt'
    # models_path = '/home/jlai/study/Tensorflow_Model_Slim_Classify/tmp/checkpoints/resnet_v1_50.ckpt'
    # models_path = '/home/jlai/study/Tensorflow_Model_Slim_Classify/tmp/flowers-models/resnet_v1_50/model.ckpt-3001'
    # models_path = '/home/jlai/study/Tensorflow_Model_Slim_Classify/tmp/resnet_v1_50-models/model.ckpt-10000'
    # models_path = 'models/0423_1/best_models_37500_0.9158.ckpt'
    models_path = '/home/jlai/study/tensorflow_models_learning/models/resnet_50/best_models_13400_0.9321.ckpt'
    # models_path = '/home/jlai/study/tensorflow_models_learning/models/resnet_50_wu/model.ckpt-10'

    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths = 3
    data_format = [batch_size, resize_height, resize_width, depths]
    predict(models_path, image_dir, labels_filename, class_nums, data_format)
