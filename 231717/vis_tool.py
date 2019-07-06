# coding=utf-8

import cv2
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 100000000000


anno_map = Image.open('image_3_predict.png')
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
cv2.imwrite('../data/vis/image_3_predict.png', anno_vis)
