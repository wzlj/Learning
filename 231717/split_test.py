# 分割遥感图片
# from osgeo import gdal
import numpy as np
from PIL import Image
import cv2
import os

np.set_printoptions(threshold=np.inf) #print not show ...
Image.MAX_IMAGE_PIXELS = 100000000000
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=np.nan)
# 要拆分的图片大小
split_width = 512
split_height = 512

# 图片所在路径
data_dir = "../data/test/src"
files = os.listdir(data_dir)

# 图片保存路径
save_path = "../data/test/data512"
if not os.path.exists(save_path):
	os.mkdir(save_path)

for img_file in files:
    # 去掉图片的后缀名作为图片保存的文件夹
	file_base = img_file[0:len(img_file)-4]

	print("\n\n ----------------- ", file_base)
	save_file_dir = os.path.join(save_path)
	if not os.path.exists(save_file_dir):
		os.mkdir(save_file_dir)

	file_path = os.path.join(data_dir, img_file)
	# dataset = gdal.Open(file_path)
	# rows = dataset.RasterXSize
	# cols = dataset.RasterYSize
	# description = dataset.GetDescription()
	# raster_count = dataset.RasterCount
	# print("file name = {}, description = {}, raster count = {}, cols = {}, rows = {}".format(file_path, description, raster_count, cols, rows))

	img = Image.open(file_path)
	img = np.asanyarray(img)
	width, height, _ = img.shape
	split_row_count = int(height/split_width) + 1
	split_col_count = int(width/split_height) + 1

    # 获取RGBA通道，因为A通道都是255感觉没用，所以不保存A通道
	# print("row = {}, row count = {}".format(rows, split_row_count))
	# band_r = dataset.GetRasterBand(3)
	# band_g = dataset.GetRasterBand(2)
	# band_b = dataset.GetRasterBand(1)

	for i in range(split_row_count):
		print("row index = {}, count = {}".format(i, split_row_count))
		for j in range(split_col_count):
			_img = np.zeros((split_width, split_height, 3)).astype(np.int32)
			print("_img shape ", _img.shape)
			x1 = j * split_width
			y1 = i * split_height
			x2 = x1 + split_width
			y2 = y1 + split_height
			if i == split_row_count - 1:
				y2 = height - y1
			if j == split_col_count - 1:
				x2 = width - x1

			print("\n\n -------------  ", x1, x2, y1, y2)
			# if (img[:, :, 3] == 255).sum() < split_height * split_width:
			if (img[x1:x2, y1:y2, :3] == 0).sum() < split_height * split_width * 3 - 50 * 3:
				print("\n\n", x2-x1,_img[x1:x2, y1:y2, :].shape)
				_img[:x2-x1, :y2-y1, :] = img[x1:x2, y1:y2, :3]
				# print("column index  = {}, count = {}, x offset = {}, yoffset = {}, width = {}, height = {}".format(j, split_col_count, xoffset, yoffset, width, height))
				_img = Image.fromarray(np.uint8(_img))
				_img.save('../data/test/data512/' + file_base + '_{}_{}.png'.format(i, j))

	# r = band_r.ReadAsArray(xoffset, yoffset, width, height)
			# g = band_g.ReadAsArray(xoffset, yoffset, width, height)
			# b = band_b.ReadAsArray(xoffset, yoffset, width, height)
			# img = cv2.merge([r, g, b])
			# file_save_path = os.path.join(save_file_dir, "{}_{}_{}.jpg".format(file_base[len("image_"):], i, j))
			# cv2.imwrite(file_save_path, img)
