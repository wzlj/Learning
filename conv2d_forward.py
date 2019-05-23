import tensorflow as tf
import numpy as np
import math

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (0, 0)),
                   'constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z


def np_conv2d_forward(A_prev, W,  stride=(1, 2, 2, 1), padding="SAME"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f_h, f_w, n_C_prev, n_C) = W.shape
    if padding == "SAME":
        # pad = 0
        n_H = math.ceil(n_H_prev/stride[1])
        n_W = math.ceil(n_W_prev/stride[2])
        pad_needed_h = (n_H - 1) * stride[1] + f_h - n_H_prev
        pad_needed_w = (n_W - 1) * stride[1] + f_w - n_W_prev
        pad_t = pad_needed_h // 2
        pad_b = pad_needed_h - pad_t
        pad_l = pad_needed_w // 2
        pad_r = pad_needed_w - pad_l

        A_prev_pad = zero_pad(A_prev, (pad_t, pad_b, pad_l, pad_r))

    else:
        n_H = math.ceil((n_H_prev - f_h + float(1.0)) / stride[1])
        n_W = math.ceil((n_W_prev - f_w + float(1.0)) / stride[2])
        A_prev_pad = A_prev

    Z = np.zeros((m, n_H, n_W, n_C))
    # print((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride[1]
                    vert_end = vert_start + f_h
                    horiz_start = w * stride[2]
                    horiz_end = horiz_start + f_w
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # print(a_slice_prev.shape, W[:, :, :, c].shape)
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c]))
                        # conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    return Z

def cpu_conv2d(X, kernel, strides, padding="SAME"):
    with tf.device('/cpu:0'):
        data = tf.constant(X, dtype=tf.float32)
        weight = tf.constant(kernel, dtype=tf.float32)
        ret_op = tf.nn.conv2d(data, weight,  strides=strides, padding=padding)
        with tf.Session() as sess:
            result = sess.run(ret_op)
            return result


def pint_conv2d(X, kernel, strides, padding="SAME"):
    with tf.device('/gpu:0'):
        data = tf.constant(X, dtype=tf.float32)
        weight = tf.constant(kernel, dtype=tf.float32)
        ret_op = tf.nn.conv2d(data, weight,  strides=strides, padding=padding)
        with tf.Session() as sess:
            result = sess.run(ret_op)
            return result


def sdk_conv2d():
    m_shape = (2, 14, 14, 512)
    a_np = np.random.normal(-5, 5, size=m_shape).astype(np.float32)

    ksize = (3, 3, 512, 3)
    weight = np.random.normal(-3, 3, size=ksize).astype(np.float32)

    strides = (1, 1, 1, 1)

    pint_ret = pint_conv2d(a_np, weight, strides, 'VALID')
    ref_ret = np_conv2d_forward(a_np, weight, strides, 'VALID')
    # ref_ret = cpu_conv2d(a_np, weight, strides, 'SAME')

    mse = np.mean(np.square(pint_ret - ref_ret))
    print("For bias add, the mse is %f" % mse)
    assert mse < 1e-8, "MSE of conv2d_forward is out of range!"

    return 0


if __name__ == "__main__":
    sdk_conv2d()
    # np.random.seed(1)
    # a_slice_prev = np.random.randn(4, 4, 3)
    # W = np.random.randn(4, 4, 3)
    # b = np.random.randn(1, 1, 1)
    # Z = conv_single_step(a_slice_prev, W, b)
    # print("Z =", Z)
