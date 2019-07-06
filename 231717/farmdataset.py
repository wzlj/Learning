from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
# from osgeo import gdal
from torchvision import transforms
import glob
import torch as tc
import numpy as np


class FarmDataset(Dataset):
    def __init__(self, istrain=True, isaug=True):
        self.istrain = istrain
        self.trainxformat = '../data/train/data512/*.png'
        self.trainyformat = '../data/train/label512/*.png'
        self.testxformat = '../data/test/*.png'
        self.fns = glob.glob(self.trainxformat) if istrain else glob.glob(self.testxformat)
        self.length = len(self.fns)
        self.transforms = transforms
        self.isaug = isaug

    def __len__(self):
        # total length is 2217
        return self.length

    def __getitem__(self, idx):
        if self.istrain:

            imgxname = self.fns[idx]
            sampleimg = Image.open(imgxname)
            imgyname = imgxname.replace('data512', 'label512')
            targetimg = Image.open(imgyname).convert('L')
            # sampleimg.save('original.bmp')

            # data augmentation
            if self.isaug:
                sampleimg, targetimg = self.imgtrans(sampleimg, targetimg)

            # check the result of dataautmentation
            # sampleimg.save('sampletmp.bmp')
            # targetimg.save('targettmp.bmp')

            sampleimg = transforms.ToTensor()(sampleimg)
            # targetimg=transforms.ToTensor()(targetimg).squeeze(0).long()
            targetimg = np.array(targetimg)
            targetimg = tc.from_numpy(targetimg).long()  # to tensor
            # print(sampleimg.shape,targetimg.shape)
            return sampleimg, targetimg
        else:
            imgxname = self.fns[idx]
            print("\n\n =============> % \n\n" %self.fns)
            return Image.open(imgxname)

    def imgtrans(self, x, y, outsize=412):
        '''input is a PIL image
           image dataaugumentation
           return also aPIL image。
        '''
        # rotate should consider y
        degree = np.random.randint(360)
        # x = x.rotate(degree, resample=Image.NEAREST, fillcolor=0)
        # y = y.rotate(degree, resample=Image.NEAREST, fillcolor=0)  # here should be carefull, in case of label damage
        x = x.rotate(degree, resample=Image.NEAREST)
        y = y.rotate(degree, resample=Image.NEAREST)  # here should be carefull, in case of label damage
        # random do the input image augmentation
        if np.random.random() > 0.5:
            # sharpness
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Sharpness(x)
            x = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # color augument
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Color(x)
            x = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # contrast augument
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Contrast(x)
            x = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # brightness
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Brightness(x)
            x = enhancer.enhance(factor)

        # img flip
        transtypes = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                      Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        transtype = transtypes[np.random.randint(len(transtypes))]
        x = x.transpose(transtype)
        y = y.transpose(transtype)


        # discard process of resize and crop
        # # img resize between 0.8-1.2
        # w, h = x.size
        # factor = 1 + np.random.normal() / 5
        # if factor > 1.2:
        #     factor = 1.2
        # if factor < 0.8:
        #     factor = 0.8
        # # print(factor,x.size)
        # x = x.resize((int(w * factor), int(h * factor)), Image.NEAREST)
        # y = y.resize((int(w * factor), int(h * factor)), Image.NEAREST)
        #
        # # random crop
        # w, h = x.size
        # stx = np.random.randint(w - outsize)
        # sty = np.random.randint(h - outsize)
        # # print((stx,sty,outsize,outsize))
        # x = x.crop((stx, sty, stx + outsize, sty + outsize))  # stx,sty,width,height
        # y = y.crop((stx, sty, stx + outsize, sty + outsize))
        # print(x.size,y.size)
        return x, y  # return outsized pil image


if __name__ == '__main__':
    d = FarmDataset(istrain=True)
    x, y = d[216]
    print(x.shape)
    print(y.shape)
