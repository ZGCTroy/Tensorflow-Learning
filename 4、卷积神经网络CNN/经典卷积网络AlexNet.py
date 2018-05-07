#!~/tensorflow/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import math
import time
import tensorflow as tf
import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 禁用GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# Hyper parameters
batch_size = 32
num_batches = 100

# 一个用来显示网络每一层结构的函数，展示输出tensor的尺寸


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []

    # 第一层的卷积层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [11, 11, 3, 64],
                dtype=tf.float32,
                stddev=0.1
            ),
            name='weights'
        )
        conv = tf.nn.conv2d(
            images,
            kernel,
            [1, 4, 4, 1],
            padding='SAME'
        )
        biases = tf.Variable(
            tf.constant(
                0.0,
                shape=[64],
                dtype=tf.float32
            ),
            trainable=True,
            name='biases'
        )
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv1)
    # 第一层的LRN
    lrn1 = tf.nn.lrn(
        conv1,
        4,
        bias=1.0,
        alpha=0.001 / 9,
        beta=0.75,
        name='lrn1'
    )
    # 第一层的最大池化层
    pool1 = tf.nn.max_pool(
        lrn1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool1'
    )
    print_activations(pool1)

    # 第二层的卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [5, 5, 64, 192],
                dtype=tf.float32,
                stddev=1e-1
            ),
            name='weights'
        )
        conv = tf.nn.conv2d(
            pool1,
            kernel,
            [1, 1, 1, 1],
            padding='SAME'
        )
        biases = tf.Variable(
            tf.constant(
                0.0,
                shape=[192],
                dtype=tf.float32
            ),
            trainable=True,
            name='biases'
        )
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)
    # 第二层的LRN
    lrn2 = tf.nn.lrn(
        conv2,
        4,
        bias=1.0,
        alpha=0.001 / 9,
        beta=0.75,
        name='lrn2'
    )
    # 第二层的最大池化层
    pool2 = tf.nn.max_pool(
        lrn2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool2'
    )
    print_activations(pool2)

    # 第三层的卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [3, 3, 192, 384],
                dtype=tf.float32,
                stddev=1e-1
            ),
            name='weights'
        )
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(
                0.0,
                shape=[384],
                dtype=tf.float32
            ),
            trainable=True,
            name='biases'
        )
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # 第四层的卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [3, 3, 384, 256],
                dtype=tf.float32,
                stddev=1e-1
            ),
            name='weights'
        )
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(
                0.0,
                shape=[256],
                dtype=tf.float32,
            ),
            trainable=True,
            name='biases'
        )
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # 第五层的卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [3, 3, 256, 256],
                dtype=tf.float32,
                stddev=1e-1
            ),
            name='weights'
        )
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(
            tf.constant(
                0.0,
                shape=[256],
                dtype=tf.float32
            ),
            trainable=True,
            name='biases'
        )
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
    # 第五层的最大池化层
    pool5 = tf.nn.max_pool(
        conv5,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool5'
    )
    print_activations(pool5)

    return pool5, parameters


# 评估每轮计算时间的函数
def time_tensorflow_run(session, target, info_string):
    # 预热轮数
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        session.run(target)
        duration = time.time()-start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print(
                    '%s: step %d, duration = %.3f'
                    % (datetime.now(), i-num_steps_burn_in, duration)
                )
            total_duration += duration
            total_duration_squared += duration*duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches-mn*mn
    sd = math.sqrt(vr)
    print(
        '%s: %s across %d steps,%.3f +/- %.3f sec /batch' %
        (datetime.now(), info_string, num_batches, mn, sd)
    )


# 主函数
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(
            tf.random_normal(
                [batch_size, image_size, image_size, 3],
                dtype=tf.float32,
                stddev=1e-1
            )
        )
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    time_tensorflow_run(sess, pool5, "Forward")

    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective, parameters)

    time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()
