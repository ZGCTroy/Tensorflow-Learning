#!~/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# Just disables the warning, doesn't enable AVX/FMA
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer(
        [os.path.dirname(__file__)+"/"+file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv会将字符串（文本行)转换到具有指定默认值的由张量列构成的元组中
    # 它还会为每一列设置数据类型
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # 实际上会读取一个文件，并加载一个张量中的batch_size行
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size*50,
                                  min_after_dequeue=batch_size)


# 对数几率回归相同的参数和变量初始化
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


# 之前的推断现在用于值的合并
def combine_inputs(X):
    return tf.matmul(X, W) + b


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(
            100,
            "train.csv",
            [[0.0], [0.0], [0], [""], [""], [0.0], [
                0.0], [0.0], [""], [0.0], [""], [""]]
        )

    # 转换属性数据
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # 最终将所有特征排列在一个矩阵中，然后对该矩阵转置，使其每行对应一个样本，每列对应一种特征
    features = tf.transpose(
        tf.stack([is_first_class, is_second_class, is_third_class, gender, age])
    )
    survived = tf.reshape(survived, [100, 1])

    return features, survived


# 新的推断值是将sigmoid函数运用到前面的合并值的输出
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    # 对训练得到的模型进行评估
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


# 在一个会话对象中启动数据流图，搭建流程
with tf.Session() as sess:
    # 模型设置
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    initial_step = 0

    # 验证之前是否已经保存了检查点文件
    FilePath = os.path.dirname(__file__)+'/Model/my_model_对数几率回归'
    ckpt = tf.train.get_checkpoint_state(FilePath)
    if ckpt and ckpt.model_checkpoint_path:
        # 从检查点恢复模型参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    # 实际的训练闭环
    training_steps = 5000
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % 1000 == 0:
            print(step, sess.run([total_loss]))
            # saver.save(
            #     sess, '/home/zgctroy/Desktop/vscode/Tensorflow_Python/Model/my_model_对数几率回归/model', global_step=step)

    # 模型评估
    # saver.save(sess, '/home/zgctroy/Desktop/vscode/Tensorflow_Python/Model/my_model_对数几率回归/model',
    #           global_step=training_steps)

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
