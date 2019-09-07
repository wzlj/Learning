import tensorflow as tf
from tensorflow.python.framework import tensor_util
import numpy as np
import pandas as pd
# import json
import os
import argparse
import glob


def get_flops(row_data):
    op = row_data['op']
    isize = row_data['input']
    kszie = row_data['ksize']
    strides = row_data['strides']
    osize = row_data['output']
    # print(row_data)
    if op in ['Conv2D', 'DepthwiseConv2dNative', 'BiasAdd', 'Relu', 'Relu6', 'FusedBatchNorm',
              'Add', 'Pad', 'MaxPool', 'AvgPool', 'Mean']:
        total_size = 1
        for size in osize:
            total_size = total_size * size

    if op in ['Conv2D', 'DepthwiseConv2dNative']:
        flops = osize[0] * osize[1] * kszie[0] * kszie[1] * kszie[2] * kszie[3]

    elif op in ['BiasAdd', 'Relu', 'Relu6']:
        flops = total_size * 3.85 / 40

    elif 'FusedBatchNorm' == op:
        flops = total_size * 6.15 / 40

    elif 'Add' == op:
        flops = total_size * 4.03 / 40

    elif 'Pad' == op:
        flops = total_size

    elif op in ['MaxPool', 'AvgPool']:
        flops = total_size * 1.67
        if 'AvgPool' == op:
            flops = flops * 8 / 40
        else:
            flops = flops * 6 / 40

    elif 'MatMul' == op:
        if len(isize) == 3:
            flops = (isize[0] * isize[1] * isize[2]) * osize[0]
        else:
            flops = (isize[-1]) * osize[0]

    elif 'Softmax' == op:
        flops = isize[0] * 54556 / 1000
        flops = flops / 40 + 3000
    elif 'Mean' == op:
        flops = total_size + 3000
    else:
        flops = 3000

    if flops < 1000:
        flops = 1000

    return int(flops)


def get_op_time(row, freq=1.0, usage=0.4, scale=0.4, fuse=False):
    op = row['op']
    flops = row['flops/cycles']

    if op in ['MatMul', 'Conv2D', 'DepthwiseConv2dNative']:
        real = 8 * 8 * 32 * freq * 10 ** 6 * usage
        time = flops * 2 / real
        # ksize = row['ksize']
        if 'Conv2D' == op:
            # print(row)
            ksize = row['ksize']
            if ksize[:2] == [1, 1]:
                time = time * 2
                # print("1x1")

        if 'DepthwiseConv2dNative' == op:
            # print("{} {} before".format(op, time))
            time = time * 4
            # print("{} {} after".format(op, time))


    elif op in ['BiasAdd', 'Relu', 'Relu6', 'FusedBatchNorm']:
        if fuse:
            time = flops / (freq * 10 ** 6) * scale
        else:
            time = flops / (freq * 10 ** 6)
    else:
        time = flops / (freq * 10 ** 6)

    return time


def get_ops_info(ops_info, tf_op_list, tensors, columns):
    for op in tf_op_list:
        row = {}
        op_type = op.type
        inputs = op.node_def.input
        strides = []
        ksize = []
        if op_type in ['Conv2D', 'DepthwiseConv2dNative']:
            # op
            row[columns[0]] = op_type
            # input
            input_ = inputs[0]
            shape = tensors[input_][1:]
            row[columns[1]] = shape
            # ksize(weights)
            input_ = inputs[1]
            shape = tensors[input_]
            row[columns[2]] = shape
            # strides
            _strides = op.node_def.attr['strides']
            strides = []
            for i in _strides.list.i:
                strides.append(i)
            row[columns[3]] = strides
            row[columns[4]] = tensors[op.node_def.name][1:]
            # output
            ops_info = ops_info.append(row, ignore_index=True)
        elif op_type in ['MaxPool', 'AvgPool']:
            # op
            row[columns[0]] = op_type
            # input
            input_ = inputs[0]
            shape = tensors[input_][1:]
            row[columns[1]] = shape
            # ksize(weights)
            _ksize = op.node_def.attr['ksize']
            for i in _ksize.list.i:
                ksize.append(i)
            row[columns[2]] = ksize
            # strides
            _strides = op.node_def.attr['strides']
            for i in _strides.list.i:
                strides.append(i)
            row[columns[3]] = strides
            # output
            row[columns[4]] = tensors[op.node_def.name][1:]
            ops_info = ops_info.append(pd.Series(row), ignore_index=True)
        elif op_type in ['BiasAdd', 'Add', 'Relu', 'Relu6', 'Squeeze',
                         'Softmax', 'FusedBatchNorm', 'Mean', 'Pad']:
            # op
            row[columns[0]] = op_type
            # input
            input_ = inputs[0]
            shape = tensors[input_][1:]
            row[columns[1]] = shape
            # ksize(weights)
            row[columns[2]] = ksize
            # strides
            row[columns[3]] = strides
            # output
            row[columns[4]] = tensors[op.node_def.name][1:]
            ops_info = ops_info.append(pd.Series(row), ignore_index=True)

        elif op_type in ['MatMul', ]:
            # op
            row[columns[0]] = op_type
            # input
            input_ = inputs[1]
            shape = tensors[input_]
            row[columns[1]] = [shape[0], ]
            # ksize(weights)
            row[columns[2]] = shape
            # strides
            row[columns[3]] = strides
            # output
            row[columns[4]] = tensors[op.node_def.name][1:]
            ops_info = ops_info.append(pd.Series(row), ignore_index=True)

            # print(row)
            # if op_type == 'MatMul':
            #     print("inputs : \n{}".format(inputs))
            #     for input_ in inputs:
            #         print(tensors[input_])
            #     print(op.node_def.name, tensors[op.node_def.name][1:])
            #     print('*'*80)
            #     print(tensors)

        elif op_type in ["ConcatV2"]:
            # op
            row[columns[0]] = op_type
            # input
            row[columns[1]] = tensors[op.node_def.name][1:]
            # ksize(weights)
            row[columns[2]] = ksize
            # strides
            row[columns[3]] = strides
            # output
            row[columns[4]] = tensors[op.node_def.name][1:]
            ops_info = ops_info.append(pd.Series(row), ignore_index=True)

    return ops_info


def gen_performance_file(args):
    path = args.pb_dir
    usage_max = args.usage_max
    usage_min = args.usage_min
    scale_max = args.scale_max
    scale_min = args.scale_min
    freq = args.frequency

    pb_path = os.path.join(path, '*.pb')
    pb_files = glob.glob(pb_path)
    xls_path = 'xls'
    if not os.path.exists(xls_path):
        os.makedirs(xls_path)
    performance_path = os.path.join(xls_path, 'performance.xls')

    performance_info = pd.DataFrame(columns=['network', 'No fused min', 'No fused max', 'Fused min', 'Fused max'])
    for file in pb_files:
        # print(file)
        prefix_name = os.path.splitext(file)[0]
        network_name = prefix_name.split('/')[-1]
        network_xls = os.path.join(xls_path, network_name+'.xls')

        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(file, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
            tf_op_list = graph.get_operations()  # used to save extracted tf ops.

        tensors = {}
        for op in tf_op_list:
            row = []
            shape = op.values()[0].shape  # op.values()[0].get_shape()
            tensors[op.node_def.name] = shape.as_list()

        ops_info = pd.DataFrame(columns=['op', 'input', 'ksize', 'strides', 'output'])
        columns = ['op', 'input', 'ksize', 'strides', 'output']

        ops_info = get_ops_info(ops_info, tf_op_list, tensors, columns)
        print("\n\nops_info of {} is created.".format(prefix_name))
        # print("#" * 120)
        # print(ops_info)

        ops_info['flops/cycles'] = ops_info.apply(get_flops, axis=1)
        ops_info['No fused max'] = ops_info.apply(get_op_time, axis=1, freq=freq, usage=usage_min, fuse=False)
        ops_info['No fused min'] = ops_info.apply(get_op_time, axis=1, freq=freq, usage=usage_max, fuse=False)
        ops_info['Fused max'] = ops_info.apply(get_op_time, axis=1, freq=freq, usage=usage_min, scale=scale_max, fuse=True)
        ops_info['Fused min'] = ops_info.apply(get_op_time, axis=1, freq=freq, usage=usage_max, scale=scale_min, fuse=True)
        print("\n\noperation time is calculated.")
        print("#" * 120)
        print(ops_info)
        ops_info.to_excel(network_xls, encoding='utf-8', index=False)
        print("\n\n{} is saved.".format(network_xls))

        row = {}
        row['network'] = network_name
        row['No fused min'] = sum(ops_info['No fused min'])
        row['No fused max'] = sum(ops_info['No fused max'])
        row['Fused max'] = sum(ops_info['Fused max'])
        row['Fused min'] = sum(ops_info['Fused min'])
        performance_info = performance_info.append(row, ignore_index=True)

    performance_info = performance_info.sort_values(by='network', ascending=True)

    performance_info.to_excel(performance_path, encoding='utf-8', index=False)
    print(performance_info)
    print("\n\n{} is saved.".format(performance_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input pb name')

    parser.add_argument('--pb_dir',
                        type=str,
                        help='The path of pb files',
                        default='/dataset/pbs/')
    parser.add_argument('--frequency',
                        type=float,
                        help='The frequency of device, GHz',
                        default=1)
    parser.add_argument('--usage_max',
                        type=float,
                        help='The max usage of device',
                        default=0.6)
    parser.add_argument('--usage_min',
                        type=float,
                        help='The min usage of device',
                        default=0.4)
    parser.add_argument('--scale_max',
                        type=float,
                        help='the max ratio of operation fused to no fused ',
                        default=0.6)
    parser.add_argument('--scale_min',
                        type=float,
                        help='the min ratio of operation fused to no fused ',
                        default=0.4)

    args = parser.parse_args()

    gen_performance_file(args)
