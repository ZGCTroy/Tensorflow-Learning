

tf.summary.scalar("output_summary", output)
merged_summaries = tf.summary.merge_all()

with tf.Session() as sess:
    # 模型设置
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)
    display_step=1000
    initial_step = 0
    # 从检测点恢复模型
    FilePath = os.path.dirname(__file__) + '/Model' + '/model'
    ckpt = tf.train.get_checkpoint_state(FilePath)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    # 实际的训练闭环
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % display_step == 0:
            print(step, sess.run([loss]))
            saver.save(
                sess,
                FilePath,
                global_step=step
            )
            writer = tf.summary.FileWriter('./Model/improved_graph', graph)
            writer.add_summary(summary, global_step=step)


    evaluate(sess, X, Y)
    writer.flush()
    writer.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()
