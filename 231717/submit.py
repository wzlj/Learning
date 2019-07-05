from farmlanddataset import FarmDataset
import torch as tc
# from osgeo import gdal
from torchvision import transforms
import png
import numpy as np

use_cuda = True
model = tc.load('./tmp/model30')  # torch.save(model,'./tmp/model{}'.format(epoch))
device = tc.device("cuda" if use_cuda else "cpu")
model = model.to(device)
model.eval()
ds = FarmDataset(istrain=False)


def createres(d, outputname):
    # 创建一个和ds大小相同的灰度图像BMP
    driver = gdal.GetDriverByName("BMP")
    # driver=ds.GetDriver()
    od = driver.Create('./tmp/' + outputname, d.RasterXSize, d.RasterYSize, 1)
    return od


def createpng(height, width, data, outputname):
    w = png.Writer(width, height, bitdepth=2, greyscale=True)
    of = open('./tmp/' + outputname, 'wb')
    w.write_array(of, data.flat)
    of.close()
    return


def predict(d, outputname='tmp.bmp'):
    wx = d.RasterXSize  # width
    wy = d.RasterYSize  # height
    print(wx, wy)
    od = data = np.zeros((wy, wx), np.uint8)
    # od=createres(d,outputname=outputname)
    # ob=od.GetRasterBand(1) #得到第一个channnel
    blocksize = 1024
    step = 512
    for cy in range(step, wy - blocksize, step):
        for cx in range(step, wx - blocksize, step):
            img = d.ReadAsArray(cx - step, cy - step, blocksize, blocksize)[0:3, :, :]  # channel*h*w
            if (img.sum() == 0): continue
            x = tc.from_numpy(img / 255.0).float()
            # print(x.shape)
            x = x.unsqueeze(0).to(device)
            r = model.forward(x)
            r = tc.argmax(r.cpu()[0], 0).byte().numpy()  # 512*512
            # ob.WriteArray(r,cx,cy)
            od[cy - step // 2:cy + step // 2, cx - step // 2:cx + step // 2] = r[256:step + 256, 256:step + 256]
            print(cy, cx)
    # del od
    createpng(wy, wx, od, outputname)
    return


print("start predict.....")
predict(ds[0], 'image_3_predict.png')
print("start predict 2 .....")
predict(ds[1], 'image_4_predict.png')
