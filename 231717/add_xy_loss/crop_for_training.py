from PIL import Image
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = 100000000000
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)


def crop_image(number, size=256, ratio=0.7):
    #
    imagepath = '../data/jingwei_round1_train_20190619/image_{}.png'.format(number)
    n = os.path.basename(imagepath)[:-4]
    labelname = '../data/jingwei_round1_train_20190619/' + n + '_label.png'
    img_label = Image.open(labelname)
    img = Image.open(imagepath)

    img = np.asanyarray(img)
    img_label = np.asanyarray(img_label)
    print(img.shape)
    print(img_label.shape)

    height, width, _ = img.shape
    unit_size = size
    y1, y2, x1, x2 = 0, unit_size, 0, unit_size
    nullthresh = unit_size * unit_size * (1 - ratio)

    print(x1, x2, y1, y2)

    while (x1 < width):
        # 判断横向是否越界
        if x2 > width:
            x2, x1 = width, width - unit_size

        while y1 < height:
            if y2 > height:
                y2, y1 = height, height - unit_size

            _img = img[y1:y2, x1:x2, :3]
            _img_label = img_label[y1:y2, x1:x2]
            if (_img_label[:, :] == 0).sum() > nullthresh:
                y1 += unit_size
                y2 += unit_size
                continue

            print("working ---- x1 = %d y1 = %d " % (x1, y1))
            print("\n\n shape of _img_label is  ", _img_label.shape)
            print("\n\n shape of _img is  ", _img.shape)
            _img = Image.fromarray(np.uint8(_img))
            dst_path = '../data/train/data{}_{}/'.format(size, ratio)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            data_path = os.path.join(dst_path, n+'_{}_{}.png'.format(x1, y1))
            _img.save(data_path)

            print(_img_label.shape)
            _img_label = Image.fromarray(_img_label).convert('L')
            print(_img_label.size)
            dst_path = '../data/train/label{}_{}/'.format(size, ratio)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            label_path = os.path.join(dst_path, n+'_{}_{}.png'.format(x1, y1))
            _img_label.save(label_path)

            y1 += unit_size
            y2 += unit_size
        if x2 == width:
            break

        y1, y2 = 0, unit_size
        x1 += unit_size
        x2 += unit_size

if __name__ == '__main__':
    # for i in range(2, 3):
        # crop_image(i, size=384, ratio=0.1)
    crop_image(2, size=384, ratio=0.1)
