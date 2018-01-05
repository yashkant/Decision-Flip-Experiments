import tensorflow as tf


def model(x, logits=False, training=False):
    conv0 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', name='conv0',
                             activation=tf.nn.relu)

    pool0 = tf.layers.max_pooling2d(conv0, pool_size=[2, 2],
                                    strides=2, name='pool0')

    conv1 = tf.layers.conv2d(pool0, filters=64,
                             kernel_size=[3, 3], padding='same',
                             name='conv1', activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2],
                                    strides=2, name='pool1')

    conv2 = tf.layers.conv2d(pool1, filters=128,
                             kernel_size=[1, 1], padding='same',
                             name='conv2', activation=tf.nn.relu)

    flat = tf.reshape(conv2, [-1, 8 * 8 * 128], name='flatten')

    dense1 = tf.layers.dense(flat, units=1024, activation=tf.nn.relu,
                             name='dense1')

    dense2 = tf.layers.dense(dense1, units=128, activation=tf.nn.relu,
                             name='dense2')
    logits_ = tf.layers.dense(dense2, units=10, name='logits')  # removed dropout

    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


