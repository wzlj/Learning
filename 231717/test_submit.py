# coding=utf-8

import torch as tc
from PIL import Image
import numpy as np
import os

np.set_printoptions(threshold=np.inf) #print not show ...
Image.MAX_IMAGE_PIXELS = 100000000000

use_cuda = True
model = tc.load('./tmp/model200')
device = tc.device("cuda" if use_cuda else "cpu")
model = model.to(device)
model.eval()

step = 512

def test(model, src_path, dst_path):

    for file in  os.listdir(src_path):
        file_path = os.path.join(src_path, file)
        img = Image.open(file_path)
        img = np.asanyarray(img)
        height, width, _ = img.shape
        print(height, width)
        row_count = int(height / step) + 1
        col_count = int(width / step) + 1

        data = np.zeros((height, width), np.uint8)
        for i in range(row_count):
            for j in range(col_count):
                _img = np.zeros((step, step, 3)).astype(np.int32)
                if i == 13 and j == 50:
                    print(i, j)
                x1 = j * step
                y1 = i * step
                x2 = x1 + step
                y2 = y1 + step
                if i == row_count - 1:
                    y2 = height - y1
                if j == col_count - 1:
                    x2 = width - x1

                if (img[x1:x2, y1:y2, :3] == 0).sum() < step * step * 3 - 10 * 3:
                    _img[:y2 - y1, :x2 - x1, :] = img[y1:y2, x1:x2, :3]
                    _img = np.array(_img).transpose([2, 0, 1])
                    # predict
                    x = tc.from_numpy(_img / 255.0).float()
                    x = x.unsqueeze(0).to(device)
                    rr = model.forward(x)
                    rr = rr.detach()[0, :, :, :].cpu()
                    r = tc.argmax(rr, 0).byte().numpy()
                    data[y1:y2, x1:x2] = r[:y2 - y1, :x2 - x1]

        imgx = Image.fromarray(data).convert('L')
        imgx.save(dst_path + file.split('.')[0]+'_predict.png')


if __name__ == "__main__":
    src_path = "../data/test/src"
    dst_path = "../data/test/predict"
    test(model, src_path, dst_path)
