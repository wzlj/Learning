# coding:utf-8
import cv2
import os
from PIL import Image
import time
import copy
import numpy as np
import torch as tc


Image.MAX_IMAGE_PIXELS = 10000000000000

# 原始测试数据路径
org_test_Path1 = '../data/test/src/image_3.png'
org_test_Path2 = '../data/test/src/image_4.png'

# 测试数据切割后文件保存路径
test_Path3 = '../data/test//test_img3_split/'
if not os.path.exists(test_Path3):
	os.makedirs(test_Path3)
test_Path4 = '../data/test//test_img4_split/'
if not os.path.exists(test_Path4):
	os.makedirs(test_Path4)

# 待拼接的测试数据路径
predict_img3 = '../data/result/to_be_merged/img3'
if not os.path.exists(predict_img3):
	os.makedirs(predict_img3)
predict_img4 = '../data/result/to_be_merged/img4'
if not os.path.exists(predict_img4):
	os.makedirs(predict_img4)


# to submit
dst_path ="../data/submit/"
if not os.path.exists(dst_path):
	os.makedirs(dst_path)

dst_path3 = os.path.join(dst_path, "image3_predict.png")
dst_path4 = os.path.join(dst_path, "image4_predict.png")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 裁剪图片（截图图片的一部分）
def crop_image(FilePath, colwidth, rowheight, dstpath):
	file = os.path.split(FilePath)	 # 得到：image_1.png
	fn = file[1].split('.')
	basename = fn[0]			     # 得到：image_1
	ext = fn[-1]			         # 得到：png


	img = Image.open(FilePath)
	box = (0, 0, colwidth, rowheight)
	img.crop(box).save(os.path.join(dstpath, basename + '.' + ext), ext)

# 切割图片（大图切分成数个小图），
# 每行的最后一张分割图的大小还要加上剩余的部分的列像素
# 每列的最后一张分割图的大小还要加上剩余的部分的行像素
def split_image(FilePath, size, savepath):
	timestart = time.time()
	print('开始处理图片切割, 请稍候...')
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	# --------------------------------------------------------------------------
	Image.MAX_IMAGE_PIXELS = 100000000000
	img = Image.open(FilePath)
	Width, Height = img.size          # 得到图片的尺寸：宽、高像素
	print(FilePath)                   # 显示图片路径
	print('原图信息: %s * %s, %s, %s' % (Width, Height, img.format, img.mode))
	print('分割像素：{}*{}'.format(size, size))

	rownum = Height // size
	colnum = Width  // size
	print('分割行列：{} * {}'.format(colnum, rownum))
	# --------------------------------------------------------------------------
	file = os.path.split(FilePath)	  # 得到：'/home/bobo/614/liuq/test_data', 'image_3.png
	fn = file[1].split('.')			  # 得到：image_3和png
	basename = fn[0]	 			  # 得到图片名称：image_3
	ext = fn[-1]		 		      # 得到图片格式后缀：png

	num = 0
	for r in range(rownum):           # 行循环
		if (r == (rownum - 1)):       # 判断是否是最后1行
			rr = Height % size   # 多余的行
			print('rr={}'.format(rr))
		else:
			rr = 0

		for c in range(colnum):       # 列循环
			num = num + 1
			if (c == (colnum - 1)):   # 判断是否是最后1列
				cc = Width % size # 多余的列
				print('cc={}'.format(cc))
			else:
				cc = 0
			box = (c * size, r * size, (c + 1) * size+ cc, (r + 1) * size+ rr)  # 左上角，右下脚坐标
			print('c={} r={}'.format(c, r))
			img.crop(box).save(os.path.join(savepath, basename + '_' + str(num) + '.' + ext), ext)

	print('图片切割完毕，共生成 %s 张小图片。' % num)
	print("切割耗时：%s" % (time.time()-timestart))
	# --------------------------------------------------------------------------

# 定义图像拼接函数
def image_compose(ImagePath, src_size, width, height, saved_path):
	timestart = time.time()
	print('开始处理图片拼接, 请稍候...')
	Image_FORMAT = ['.jpg', '.JPG', '.png']  # 图片格式
	Image_SIZE = src_size  # 小图大小为512*512
	Image_Col = width // Image_SIZE     # 图片间隔，也就是合并成一张图后，一共有几列
	Image_Row = height // Image_SIZE     # 图片间隔，也就是合并成一张图后，一共有几行

	# 获取图片文件夹下的所有图片名称
	image_names = [name for name in os.listdir(ImagePath) for item in Image_FORMAT if
				   os.path.splitext(name)[1] == item]
	image_names = sorted(image_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))

	to_image = Image.new('L', (width, height))  # 创建一个新图,不要创建RGB的

	# 循环遍历，把每张图片按顺序粘贴到对应位置上
	to_image.MAX_IMAGE_PIXELS = 100000000000
	for y in range(Image_Row):
		for x in range(Image_Col):
			from_image = Image.open(os.path.join(ImagePath, image_names[Image_Col * y + x]))
			to_image.paste(from_image, ((x * Image_SIZE), (y * Image_SIZE)))
	to_image.save(saved_path)  # 保存新图
	print("图像拼接耗时：%s" % (time.time()-timestart))


def test(model, src_path, dst_path):

	imagelist = os.listdir(src_path)
	imgNum = int(len(imagelist))
	imagelist.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

	for i in range(imgNum):
		img = Image.open(os.path.join(src_path, imagelist[i]))  # 读取图片
		img = np.asanyarray(img)
		height, width, _ = img.shape  # img.shape[0]：图像的垂直尺寸（高度） img.shape[1]：图像的水平尺寸（宽度） img.shape[2]：图像的通道数
		print(img.shape)

		_img = np.zeros((height, width, 3)).astype(np.int32)
		_img[:, :, :] = img[:, :, :3]  # 只取前面三个通道数据：RGB

		_img = np.array(_img).transpose([2, 0, 1])  # 维度：C*H*W
		print(img.shape)

		# predict
		x = tc.from_numpy(_img / 255.0).float()  # 归一化数据
		x = x.unsqueeze(0).to(device)  # 第0维设置为1最后形成维度：1*C*H*W   --增加一个维度
		rr = model.forward(x)
		rr = rr.detach()[0, :, :, :].cpu()  # 表示不回传梯度
		data = tc.argmax(rr, 0).byte().numpy()  # 指定dim=0时，行的size没有了。求每一列的最大行标！

		imgx = Image.fromarray(data).convert('L')  # PIL的九种不同模式：1，L，P，RGB，RGBA，CMYK，YCbCr,I，F
		predict_path = os.path.join(os.path.join(dst_path, str(i) + '.png'))
		imgx.save(predict_path)


def visualize(src_img, dst_img):
	anno_map = Image.open(src_img)
	anno_map = np.asarray(anno_map)
	print(anno_map.shape)
	B = anno_map.copy()   # 蓝色通道
	B[B == 1] = 255
	B[B == 2] = 0
	B[B == 3] = 0

	G = anno_map.copy()   # 绿色通道

	G[G == 1] = 0
	G[G == 2] = 255
	G[G == 3] = 0

	R = anno_map.copy()   # 红色通道

	R[R == 1] = 0
	R[R == 2] = 0
	R[R == 3] = 255

	anno_vis = np.dstack((B, G, R))
	anno_vis = cv2.resize(anno_vis, None, fx=0.1, fy=0.1)
	cv2.imwrite(dst_img, anno_vis)

	print("visulization of {} is done".format(src_img))


if __name__ == "__main__":
	print( "----数据预处理任务开始------")
	start = time.time()
	split = False

	use_cuda = True
	model = tc.load('../data/models/model_data1024_0.1/72')
	device = tc.device("cuda" if use_cuda else "cpu")
	model = model.to(device)
	model.eval()


	# ==========================================================================
	if split:
		split_image(org_test_Path1, 1024, test_Path3)    # 分成共生成 74 * 39=2886 张小图片，耗时：316.13400530815125
		split_image(org_test_Path2, 1024, test_Path4)  # 分成共生成 51 * 57=2907 张小图片，耗时：304.5255982875824
	else:
		#
		# 4 :(28832, 25936)
		# 3:(19903, 37241)

		test(model, test_Path3, predict_img3)
		image_compose(predict_img3, 1024, 37241, 19903, dst_path3)

		test(model, test_Path4, predict_img4)
		image_compose(predict_img4, 1024, 25936, 28832, dst_path4)

		vis_path = os.path.join('../data', 'vis_debug', dst_path.split('/')[-1])
		if not os.path.exists(vis_path):
			os.makedirs(vis_path)

		for file in os.listdir(dst_path):
			src_img = os.path.join(dst_path, file)
			dst_img = os.path.join(vis_path, file)
			visualize(src_img, dst_img)
	# ==========================================================================

