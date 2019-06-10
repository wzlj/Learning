# Faster R-CNN

This version is from <https://github.com/endernewton/tf-faster-rcnn>

## ROI_POOLING

In most faster rcnn in tensorflow, roi_pooling is not implemented, it is replaced by crop_and_resize.  in line 134 from network.py 

tf.image.roi_pooling is not implemented.
