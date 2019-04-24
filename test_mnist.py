# conding=utf-8

# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import read_data

mnist = read_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 4, 4, 1], padding='SAME')


# placeholder for size of input data
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# input channels is 1, output channels is 32
w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 50], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[50]))
h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)

w_fc1 = tf.Variable(tf.truncated_normal([7*7*50, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_conv1, [-1, 7*7*50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2)+b_fc2)

loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

for i in range(10):
    batch = mnist.train.next_batch(50)
    if i % 1 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print ("step %d,train_accuracy= %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        print ("test_accuracy= %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
