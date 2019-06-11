# Faster R-CNN

This version is from <https://github.com/endernewton/tf-faster-rcnn>

## 1. ROI_POOLING

In most faster rcnn in tensorflow, roi_pooling is not implemented, it is replaced by crop_and_resize.  in line 134 from network.py 

tf.image.roi_pooling is not implemented.

## 2 architecture
The main class is NetWork implemented in tf-faster-rcnn/lib/nets/network.py
Then different backbones inherit from Network to apply faster r-cnn

Methods image_to_head and _head_to_tail is implemented in vgg16.


## 3. flow of inference

### 3.1 main flow 

net = vgg16()-----> net.create_architecture("TEST", 21, tag='default', anchor_scales=[8, 16, 32]) -----> saver = tf.train.Saver()  saver.restore(sess, tfmodel) -----> im_detect(sess, net, im) -----> net.test_image(sess, blobs['data'], blobs['im_info']) -----> apply nms by classes.

### 3.2 net.create_architecture

1. initialize parameters of class NetWork

    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

2. initialize parameters of model

    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer


### 3.3 net.test_imag branch 

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)


