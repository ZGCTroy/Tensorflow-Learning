import os
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper parameters
training_steps = 10000

# 初始化变量和模型参数，定义训练闭环中的运算
W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inputs():
    # 读取或生成训练数据X及其期望输出Y
    weight_age = [
        [84, 46], [73, 20], [65, 52], [70, 30], [76, 57],
        [69, 25], [63, 28], [72, 36], [79, 57], [75, 57],
        [75, 44], [27, 24], [89, 31], [65, 52], [57, 23],
        [59, 60], [69, 48], [60, 34], [79, 51], [75, 50],
        [82, 34], [59, 60], [69, 48], [60, 34], [79, 51],
        [75, 50], [82, 34], [59, 46], [57, 23], [85, 37],
        [55, 40], [63, 30]
    ]
    blood_fat_content = [
        354, 190, 405, 263, 451, 302, 288,
        385, 402, 365, 209, 290, 346, 254,
        395, 434, 220, 374, 308, 220, 311,
        181, 274, 303, 244
    ]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def inference(X):
    # 计算推断模型在数据X上的输出，并将结果返回
    return tf.matmul(X, W) + b


def loss(X, Y):
    # 依据训练数据X及其期望输出Y计算损失
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    learning_rate = 0.001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    # 对训练得到的模型进行评估
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))


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
    FilePath = os.path.dirname(__file__) + '/Model'+'/model'
    print(FilePath)
    ckpt = tf.train.get_checkpoint_state(FilePath)
    if ckpt and ckpt.model_checkpoint_path:
        # 从检查点恢复模型参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    # 实际的训练闭环
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % 1000 == 0:
            print(step, sess.run([total_loss]))
            saver.save(
                sess,
                FilePath,
                global_step=step
            )

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
