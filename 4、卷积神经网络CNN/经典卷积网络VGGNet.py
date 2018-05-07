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


def conv_op(input_op, name, kh, kw, n_out, dh, dw, parameters):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope+"w",
            shape=[kh, kw, n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        biases = tf.Variable(
            tf.constant(
                0.0,
                shape=[n_out],
                dtype=tf.float32
            ),
            trainable=True,
            name='biases'
        )
        conv = tf.nn.conv2d(
            input_op,
            kernel,
            (1, dh, dw, 1),
            padding='SAME'
        )
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope)
        parameters += [kernel, biases]
        return conv


def fc_op(input_op, name, n_out, parameters):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope+"w",
            shape=[n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        biases = tf.Variable(
            tf.constant(
                0.1,
                shape=[n_out],
                dtype=tf.float32
            ),
            name='biases'
        )
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        parameters += [kernel, biases]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(
        input_op,
        ksize=[1, kh, kw, 1],
        strides=[1, dh, dw, 1],
        padding='SAME',
        name=name
    )


def inference_op(input_op, keep_prob):
    parameters = []
    conv1_1 = conv_op(
        input_op,
        name="conv1_1",
        kh=3,
        kw=3,
        n_out=64,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv1_2 = conv_op(
        conv1_1,
        name="conv1_2",
        kh=3,
        kw=3,
        n_out=64,
        dh=1,
        dw=1,
        parameters=parameters
    )
    pool1 = mpool_op(
        conv1_2,
        name="pool1",
        kh=2,
        kw=2,
        dw=2,
        dh=2
    )
    conv2_1 = conv_op(
        pool1,
        name="conv2_1",
        kh=3,
        kw=3,
        n_out=128,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv2_2 = conv_op(
        conv2_1,
        name="conv2_2",
        kh=3,
        kw=3,
        n_out=128,
        dh=1,
        dw=1,
        parameters=parameters
    )
    pool2 = mpool_op(
        conv2_2,
        name="pool2",
        kh=2,
        kw=2,
        dw=2,
        dh=2
    )
    conv3_1 = conv_op(
        pool2,
        name="conv3_1",
        kh=3,
        kw=3,
        n_out=256,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv3_2 = conv_op(
        conv3_1,
        name="conv3_2",
        kh=3,
        kw=3,
        n_out=256,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv3_3 = conv_op(
        conv3_2,
        name="conv3_3",
        kh=3,
        kw=3,
        n_out=256,
        dh=1,
        dw=1,
        parameters=parameters
    )
    pool3 = mpool_op(
        conv3_3,
        name="pool3",
        kh=2,
        kw=2,
        dw=2,
        dh=2
    )
    conv4_1 = conv_op(
        pool3,
        name="conv4_1",
        kh=3,
        kw=3,
        n_out=512,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv4_2 = conv_op(
        conv4_1,
        name="conv4_2",
        kh=3,
        kw=3,
        n_out=512,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv4_3 = conv_op(
        conv4_2,
        name="conv4_3",
        kh=3,
        kw=3,
        n_out=512,
        dh=1,
        dw=1,
        parameters=parameters
    )
    pool4 = mpool_op(
        conv4_3,
        name="pool4",
        kh=2,
        kw=2,
        dw=2,
        dh=2
    )
    conv5_1 = conv_op(
        pool4,
        name="conv5_1",
        kh=3,
        kw=3,
        n_out=512,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv5_2 = conv_op(
        conv5_1,
        name="conv5_2",
        kh=3,
        kw=3,
        n_out=512,
        dh=1,
        dw=1,
        parameters=parameters
    )
    conv5_3 = conv_op(
        conv5_2,
        name="conv5_3",
        kh=3,
        kw=3,
        n_out=512,
        dh=1,
        dw=1,
        parameters=parameters
    )
    pool5 = mpool_op(
        conv5_3,
        name="pool5",
        kh=2,
        kw=2,
        dw=2,
        dh=2
    )
    # 扁平化
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # 全连接层
    fc6 = fc_op(resh1, name="fc6", n_out=4096, parameters=parameters)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, parameters=parameters)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, parameters=parameters)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, parameters


# 评估每轮计算时间的函数
def time_tensorflow_run(sess, target, feed, info_string):
    # 预热轮数
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        sess.run(target, feed_dict=feed)
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
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, parameters = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")

    objective = tf.nn.l2_loss(fc8)
    grad = tf.gradients(objective, parameters)

    time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


batch_size = 25
num_batches = 100
run_benchmark()
