import time
from ops import *


def resnet_v1_18(inputs,
                 num_classes=None,
                 is_training=True,
                 scope='resnet_v1_18'):
    with tf.variable_scope(scope, 'resnet_v1', [inputs]) as sc:
        ch = 64
        residual_block = resblock
        # stage 1
        x = conv(inputs, channels=ch, kernel=7, stride=2, scope='conv')
        x = max_pooling(x)

        residual_list = [2, 2, 2, 2]
        # stage 2
        for i in range(residual_list[0]):
            x = residual_block(x, channels=ch, is_training=is_training, downsample=False,
                               scope='resblock0_' + str(i))

        # stage 3
        x = residual_block(x, channels=ch * 2, is_training=is_training, downsample=True, scope='resblock1_0')
        for i in range(1, residual_list[1]):
            x = residual_block(x, channels=ch * 2, is_training=is_training, downsample=False,
                               scope='resblock1_' + str(i))

        # stage 4
        x = residual_block(x, channels=ch * 4, is_training=is_training, downsample=True, scope='resblock2_0')
        for i in range(1, residual_list[2]):
            x = residual_block(x, channels=ch * 4, is_training=is_training, downsample=False,
                               scope='resblock2_' + str(i))

        #stage 5
        x = residual_block(x, channels=ch * 8, is_training=is_training, downsample=True, scope='resblock_3_0')
        for i in range(1, residual_list[3]):
            x = residual_block(x, channels=ch * 8, is_training=is_training, downsample=False,
                               scope='resblock_3_' + str(i))

        x = batch_norm(x, is_training, scope='batch_norm')
        x = relu(x)

        x = global_avg_pooling(x)
        x = fully_conneted(x, units=num_classes, scope='logit')

        return x
