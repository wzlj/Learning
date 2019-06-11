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
          
3. apply _build_network 
        
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer, 
                        biases_initializer=tf.constant_initializer(0.0)): 
          rois, cls_prob, bbox_pred = self._build_network(training)
          
### 3.3 details of _build_network 
    
1. initializers  

        if cfg.TRAIN.TRUNCATED:
          initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
          initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
          initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
          initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

2. backbone vgg16
 
        net_conv = self._image_to_head(is_training)
    
3. apply rpn and roi_pooling
 
        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope, self._scope):
          # build the anchors for the image
          self._anchor_component()
          # region proposal network
          rois = self._region_proposal(net_conv, is_training, initializer)
          # region of interest pooling
          if cfg.POOLING_MODE == 'crop':
            pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
          else:
            raise NotImplementedError

4. flatten and fully_connected 

        fc7 = self._head_to_tail(pool5, is_training)
        
5. region classification

        with tf.variable_scope(self._scope, self._scope):
          # region classification
          cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                            initializer, initializer_bbox)

### 3.3 net.test_imag branch 

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)


