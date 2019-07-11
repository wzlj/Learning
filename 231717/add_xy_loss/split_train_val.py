# coding:utf-8

import os
import numpy as np
from PIL import Image
import random
import shutil


def split_train_val(data_path, lable_path, dst_data_path, dst_label_path, ratio=0.1):
    random.seed = 16
    np.random.seed(16)
    train_list = os.listdir(data_path)
    random.shuffle(train_list)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱
    num = int(len(train_list) * ratio)

    print(num)
    for i in range(num):
        _src_data_path = os.path.join(data_path, train_list[i])
        _dst_data_path = os.path.join(dst_data_path, train_list[i])
        shutil.move(_src_data_path, _dst_data_path)

        _lable_path = os.path.join(lable_path, train_list[i])
        _dst_label_path = os.path.join(dst_label_path, train_list[i])
        shutil.move(_lable_path, _dst_label_path)


if __name__ == "__main__":

    train_path = "../data/train"
    src_data_path = os.path.join(train_path, "data1024_0.1")
    src_label_path = os.path.join(train_path, "label1024_0.1")

    val_path = "../data/val"
    dst_data_path = os.path.join(val_path, "data1024_0.1")
    dst_label_path = os.path.join(val_path, "label1024_0.1")

    print(dst_data_path)
    print(dst_label_path)

    if not os.path.exists(dst_data_path):
        os.makedirs(dst_data_path)
    if not os.path.exists(dst_label_path):
        os.makedirs(dst_label_path)

    split_train_val(src_data_path, src_label_path, dst_data_path, dst_label_path, ratio=0.1)



