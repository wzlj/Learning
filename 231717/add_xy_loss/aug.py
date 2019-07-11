# coding:utf-8

import os
import time
import copy
import numpy as np
from PIL import Image
import random


def image_swap_LR(data_path, lable_path, aug_data_path, aug_label_path):
    random.seed = 16
    np.random.seed(16)
    train_list = os.listdir(data_path)
    random.shuffle(train_list)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱
    num = len(train_list)//2

    for i in range(num): #len(train_list)//2
        data1 = Image.open(os.path.join(data_path, train_list[i]))          # 读取图片
        data2 = Image.open(os.path.join(data_path, train_list[i + num]))
        data1 = np.asanyarray(data1)
        data2 = np.asanyarray(data2)

        height, width, _ = data1.shape    # 得到图片的尺寸：宽、高像素
        print(os.path.join(data_path, train_list[i]))
        print(os.path.join(data_path, train_list[i+1]))
        ext = train_list[i].split('.')[-1]         # 得到图片格式后缀：png

        dst_data1 = np.zeros(data1.shape)
        dst_data2 = np.zeros(data1.shape)

        dst_data1[:, :width // 2, :] = data1[:, :width // 2, :]
        dst_data1[:, width // 2:, :] = data2[:, width // 2:, :]

        dst_data2[:, :width // 2, :] = data2[:, :width // 2, :]
        dst_data2[:, width // 2:, :] = data1[:, width // 2:, :]

        print(dst_data1.shape)
        dst_data1 = Image.fromarray(np.uint8(dst_data1))
        dst_data1.save(os.path.join(aug_data_path, "{}_{}RL".format(i, i+num) + "." + ext))
        dst_data2 = Image.fromarray(np.uint8(dst_data2))
        dst_data2.save(os.path.join(aug_data_path, "{}_{}LR".format(i, i+num) + "." + ext))


        data1 = Image.open(os.path.join(lable_path, train_list[i]))          # 读取图片
        data2 = Image.open(os.path.join(lable_path, train_list[i + num]))
        data1 = np.asanyarray(data1)
        data2 = np.asanyarray(data2)

        height, width = data1.shape    # 得到图片的尺寸：宽、高像素
        print(os.path.join(lable_path, train_list[i]))
        print(os.path.join(lable_path, train_list[i+1]))
        ext = train_list[i].split('.')[-1]         # 得到图片格式后缀：png

        dst_data1 = np.zeros(data1.shape)
        dst_data2 = np.zeros(data1.shape)

        dst_data1[:, :width // 2] = data1[:, :width // 2]
        dst_data1[:, width // 2:] = data2[:, width // 2:]

        dst_data2[:, :width // 2] = data2[:, :width // 2]
        dst_data2[:, width // 2:] = data1[:, width // 2:]

        print(dst_data1.shape)
        dst_data1 = Image.fromarray(np.uint8(dst_data1))
        dst_data1.save(os.path.join(aug_label_path, "{}_{}RL".format(i, i+num) + "." + ext))
        dst_data2 = Image.fromarray(np.uint8(dst_data2))
        dst_data2.save(os.path.join(aug_label_path, "{}_{}LR".format(i, i+num) + "." + ext))



def image_swap_UD(data_path, lable_path, aug_data_path, aug_label_path):
    np.random.seed(18)
    random.seed = 18
    train_list = os.listdir(data_path)
    random.shuffle(train_list)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱
    num = len(train_list)//2

    for i in range(num): #len(train_list)//2
        data1 = Image.open(os.path.join(data_path, train_list[i]))          # 读取图片
        data2 = Image.open(os.path.join(data_path, train_list[i + num]))
        data1 = np.asanyarray(data1)
        data2 = np.asanyarray(data2)

        height, width, _ = data1.shape    # 得到图片的尺寸：宽、高像素
        print(os.path.join(data_path, train_list[i]))
        print(os.path.join(data_path, train_list[i+1]))
        ext = train_list[i].split('.')[-1]         # 得到图片格式后缀：png

        dst_data1 = np.zeros(data1.shape)
        dst_data2 = np.zeros(data1.shape)

        dst_data1[height // 2:, :, :] = data1[height // 2:, :, :]
        dst_data1[:height // 2, :, :] = data2[:height // 2, :, :]

        dst_data2[height // 2:, :, :] = data2[height // 2:, :, :]
        dst_data2[:height // 2, :, :] = data1[:height // 2, :, :]

        print(dst_data1.shape)
        dst_data1 = Image.fromarray(np.uint8(dst_data1))
        dst_data1.save(os.path.join(aug_data_path, "{}_{}UD".format(i, i+num) + "." + ext))
        dst_data2 = Image.fromarray(np.uint8(dst_data2))
        dst_data2.save(os.path.join(aug_data_path, "{}_{}DU".format(i, i+num) + "." + ext))


        data1 = Image.open(os.path.join(lable_path, train_list[i]))          # 读取图片
        data2 = Image.open(os.path.join(lable_path, train_list[i + num]))
        data1 = np.asanyarray(data1)
        data2 = np.asanyarray(data2)

        height, width = data1.shape    # 得到图片的尺寸：宽、高像素
        print(os.path.join(lable_path, train_list[i]))
        print(os.path.join(lable_path, train_list[i+1]))
        ext = train_list[i].split('.')[-1]         # 得到图片格式后缀：png

        dst_data1 = np.zeros(data1.shape)
        dst_data2 = np.zeros(data1.shape)

        dst_data1[height // 2:, :] = data1[height // 2:, :]
        dst_data1[:height // 2, :] = data2[:height // 2, :]

        dst_data2[height // 2:, :] = data2[height // 2:, :]
        dst_data2[:height // 2, :] = data1[:height // 2, :]

        print(dst_data1.shape)
        dst_data1 = Image.fromarray(np.uint8(dst_data1))
        dst_data1.save(os.path.join(aug_label_path, "{}_{}UD".format(i, i+num) + "." + ext))
        dst_data2 = Image.fromarray(np.uint8(dst_data2))
        dst_data2.save(os.path.join(aug_label_path, "{}_{}DU".format(i, i+num) + "." + ext))


if __name__ == "__main__":

    src_path = "../data/train"
    src_data_path = os.path.join(src_path, "data1024_0.1")
    src_label_path = os.path.join(src_path, "label1024_0.1")

    dst_data_path = src_data_path + "_aug"
    dst_label_path = src_label_path + "_aug"

    print(dst_data_path)
    print(dst_label_path)

    if not os.path.exists(dst_data_path):
        os.mkdir(dst_data_path)
    if not os.path.exists(dst_label_path):
        os.mkdir(dst_label_path)

    print( "----数据预处理任务开始------")
    start = time.time()
    image_swap_LR(src_data_path, src_label_path, dst_data_path, dst_label_path)      #左右对换
    image_swap_UD(src_data_path, src_label_path, dst_data_path, dst_label_path)      #上下对换
    print("处理总耗时：%s" % (time.time()-start))

