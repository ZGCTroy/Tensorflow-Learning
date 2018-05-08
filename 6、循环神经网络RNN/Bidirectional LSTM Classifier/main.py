import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Hyper parameters
learning_rate = 0.001
max_samples = 4000000
batch_size = 128
display_step = 10
n_input = 28
n_steps = 28
n_hidden = 256
n_classes = 10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_fw_cell,  # batch_size * n_hidden
        lstm_bw_cell,  # batch_size * n_hidden
        x,
        dtype=tf.float32
    )  # batch_size * 2*n_hidden
    return tf.matmul(outputs[-1], weights) + biases
    # batch_size * n_classes


pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=pred,
        labels=y
    )
)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(
    tf.argmax(pred, 1),
    tf.argmax(y, 1)
)

accuracy = tf.reduce_mean(
    tf.cast(
        correct_pred,
        tf.float32
    )
)

tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 模型设置
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./Model/Graph')
    merged_summaries = tf.summary.merge_all()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)

    # 从检测点恢复模型
    step = 1
    FilePath = os.path.dirname(__file__) + '/Model/'
    ckpt = tf.train.get_checkpoint_state(FilePath)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print("Read Model - ", step)

    # Train
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_input])
        feed_dict = {x: batch_x, y: batch_y}
        sess.run(optimizer, feed_dict=feed_dict)
        if step % display_step == 0:
            acc, loss, summary = sess.run([accuracy, cost,merged_summaries], feed_dict=feed_dict)
            saver.save(
                sess,
                FilePath,
                global_step=step*batch_size
            )
            writer.add_summary(summary, global_step=step)
            print(
                "Iter " +
                str(step * batch_size) +
                ", Minibatch Loss = " +
                "{:.6f}".format(loss) +
                ", Training Accuracy= " +
                "{:.5f}".format(acc)
            )
        step += 1

    print("Optimization Finished!")

    # Evaluation
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape([test_len, 28, 28])
    test_label = mnist.test.labels[:test_len]
    print(
        "Testing Accuracy:",
        sess.run(
            accuracy,
            feed_dict={x: test_data, y: test_label}
        )
    )

    coord.request_stop()
    coord.join(threads)
