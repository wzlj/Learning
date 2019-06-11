# Faster R-CNN

This version is from <https://github.com/endernewton/tf-faster-rcnn>

## ROI_POOLING

In most faster rcnn in tensorflow, roi_pooling is not implemented, it is replaced by crop_and_resize.  in line 134 from network.py 

tf.image.roi_pooling is not implemented.

## architecture
The main class is NetWork implemented in tf-faster-rcnn/lib/nets/network.py
Then different backbones inherit from Network to apply faster r-cnn

Methods image_to_head and _head_to_tail is implemented in vgg16.


## flow of inference

### main flow 

net = vgg16()-----> net.create_architecture("TEST", 21, tag='default', anchor_scales=[8, 16, 32]) -----> saver = tf.train.Saver()  saver.restore(sess, tfmodel) -----> im_detect(sess, net, im) -----> net.test_image(sess, blobs['data'], blobs['im_info']) -----> apply nms by classes.

### im_detect branch 




