import tensorflow as tf
import os.path
import ResNet_
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
import tensorflow.contrib.slim as slim
import SqueezeNet

labels_nums = 5  # 类别个数
batch_size = 32  #
resize_height = 224  # mobilenet_v1.default_image_size 指定存储图片高度
resize_width = 224   # mobilenet_v1.default_image_size 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

# 定义input_labels为labels数据
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def main():
    # ckpt_path = 'models/resnet18_x3_old/best_models.ckpt'
    ckpt_path = './models/SqueezeNet_07/best_models.ckpt'
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # out = ResNet_.resnet_v1_18_(inputs=input_images, num_classes=labels_nums, is_training=False)
        # with slim.arg_scope(vgg.vgg_arg_scope()):
        #     out, end_points = vgg.vgg_16(inputs=input_images, num_classes=labels_nums, is_training=False)
        out = SqueezeNet.SqueezeNet(inputs=input_images, num_classes=labels_nums)
        score = tf.nn.softmax(out, name='output')
        # tf.contrib.quantize.create_eval_graph()
        load_variables_from_checkpoint(sess, ckpt_path)
        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['output'])
        tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname('SqueezeNet.pb'),
            os.path.basename('SqueezeNet.pb'),
            as_text=False)
        tf.logging.info('Saved frozen graph to %s', 'mnist_frozen_graph.pb')


if __name__ == "__main__":
     main()
