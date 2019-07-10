# coding:utf-8
# from 刘强 四川大学614实验室
# modified by ljc
import os
import time
import copy
import numpy as np
from PIL import Image
import random


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. 图片对切分组合（图1与图片2左右部分对调）Left & Right
def image_swap_LR(train_Path, train_lablePath,save_train_Path, save_train_lablePath):
    if not os.path.exists(save_train_Path):
     os.makedirs(save_train_Path)
    if not os.path.exists(save_train_lablePath):
     os.makedirs(save_train_lablePath)

    trainlist1 = os.listdir(train_Path)
    imgNum = int(len(trainlist1)) // 2
    # trainlist1.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    #
    # trainlist2 = copy.deepcopy(trainlist1)  # 深拷贝
    random.shuffle(trainlist1)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱
    trainlist2 = trainlist1[imgNum:]
    for i in range(2):
        img1 = Image.open(os.path.join(train_Path, trainlist1[i]))          # 读取图片
        img2 = Image.open(os.path.join(train_Path, trainlist2[i]))
        Width, Height = img1.size    # 得到图片的尺寸：宽、高像素
        print(os.path.join(train_Path, trainlist1[i]))
        print(os.path.join(train_Path, trainlist2[i]))
        ext = trainlist1[i].split('.')[-1]         # 得到图片格式后缀：png

        box1 = (0, 0, int(Width/2), Height)         # 左部分
        box2 = (int(Width/2), 0, Width, Height)     # 右部分
        region1 = img1.crop(box1)    # 取出图像块左部分
        region2 = img2.crop(box2)    # 取出图像块右部分
        img1.paste(region2, box2)    # 粘贴图像块
        img2.paste(region1, box1)
        img1.save(os.path.join(save_train_Path, str(i+1+imgNum) + '.' + ext))        # 保存图像
        img2.save(os.path.join(save_train_Path, str(i+1+imgNum+imgNum) + '.' + ext))

        img3 = Image.open(os.path.join(train_lablePath, trainlist1[i]))     # 读取mask
        img4 = Image.open(os.path.join(train_lablePath, trainlist2[i]))
        print(os.path.join(train_lablePath, trainlist1[i]))
        print(os.path.join(train_lablePath, trainlist2[i]))
        region3 = img3.crop(box1)    # 取出mask图像块左部分
        region4 = img4.crop(box2)    # 取出mask图像块右部分
        img3.paste(region4, box2)    # 粘贴mask图像块
        img4.paste(region3, box1)
        img3.save(os.path.join(save_train_lablePath, str(i+1+imgNum) + '.' + ext))   # 保存图像
        img4.save(os.path.join(save_train_lablePath, str(i+1+imgNum+imgNum) + '.' + ext))
        print('图片总共 %s 张, 处理了%s 张' % (imgNum, i+1))

def image_swap_UD(data_path, lable_path, aug_data_path, aug_label_path):
    train_list = os.listdir(data_path)
    random.shuffle(train_list)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱

    for i in range(len(train_list)//2):
        data1 = Image.open(os.path.join(data_path, train_list[i]))          # 读取图片
        data2 = Image.open(os.path.join(data_path, train_list[i + 1]))
        Width, Height= data1.size    # 得到图片的尺寸：宽、高像素
        print(os.path.join(data_path, train_list[i]))
        print(os.path.join(data_path, train_list[i+1]))
        ext = train_list[i].split('.')[-1]         # 得到图片格式后缀：png

        box1 = (0, 0, Width, Height//2)         # 上部分
        box2 = (0, Height//2, Width, Height)    # 下部分
        region1 = data1.crop(box1)    # 取出图像块左部分
        region2 = data2.crop(box2)    # 取出图像块右部分
        data1.paste(region2, box2)    # 粘贴图像块
        data2.paste(region1, box1)
        data1.save(os.path.join(aug_data_path, "{}U_{}D".format(i, i+1) + "." + ext))        # 保存图像
        data2.save(os.path.join(aug_data_path, "{}D_{}U".format(i, i+1) + "." + ext))

        label1 = Image.open(os.path.join(lable_path, train_list[i]))     # 读取mask
        label2 = Image.open(os.path.join(lable_path, train_list[i]))
        print(os.path.join(lable_path, train_list[i]))
        print(os.path.join(lable_path, train_list[i+1]))
        region3 = label1.crop(box1)    # 取出mask图像块左部分
        region4 = label2.crop(box2)    # 取出mask图像块右部分
        label1.paste(region4, box2)    # 粘贴mask图像块
        label2.paste(region3, box1)
        label1.save(os.path.join(aug_label_path, "{}U_{}D".format(i, i+1) + "." + ext))   # 保存图像
        label2.save(os.path.join(aug_label_path, "{}D_{}U".format(i, i+1) + "." + ext))

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
	# image_swap_UD(src_data_path, src_label_path, dst_data_path, dst_label_path)      #上下对换
	print("处理总耗时：%s" % (time.time()-start))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
