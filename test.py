import tensorflow as tf

with tf.Session() as sess:
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    # string_input_producer can create a queue
    filename_queue = tf.train.string_input_producer(filename, num_epochs=5, shuffle=False)
    # read data from filename_queue , the method is reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # initializer epoche
    tf.local_variables_initializer().run()
    # fitting queue
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)