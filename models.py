import tensorflow as tf

n_classes = 2 # 0 or 1 for attack models

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

#     dense1 = tf.layers.dense(flat, units=1024, activation=tf.nn.relu,
#                              name='dense1')

#     dense2 = tf.layers.dense(dense1, units=128, activation=tf.nn.relu,
#                              name='dense2')

    logits_ = tf.layers.dense(flat, units=10, name='logits')

    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def sub_attack_model(model_shape):
    x = tf.placeholder(tf.float32, (None, model_shape[0]), name='x')
    y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    
    mid_layer = tf.layers.dense(x, units=model_shape[1], name='mid_layer')
    
    logit = tf.layers.dense(mid_layer, units=model_shape[2], name='logit')
    
    ybar = tf.nn.softmax(logit, name='ybar')
    
    #finding accuracy 
    z = tf.argmax(y, axis=1)
    zbar = tf.argmax(ybar, axis=1)
    count = tf.cast(tf.equal(z, zbar), tf.float32)
    acc = tf.reduce_mean(count, name='acc')
    
    #finding loss 
    xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logit)
    loss = tf.reduce_mean(xent, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    return x, y, optimizer, ybar, acc
    

