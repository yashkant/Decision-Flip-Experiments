import tensorflow as tf


def fgsm(model, x, step_size=2, clip_min=0., clip_max=1., bbox_semi_side = 0.1):
    """
    Fast gradient sign method.

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version, since the original FGSM
    has label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
 
    :return: A tensor, contains adversarial samples for each input.
    """
    x_adv = tf.identity(x)
    x_adv_llimit = tf.maximum(x_adv-x_adv, x_adv - bbox_semi_side)
    x_adv_ulimit = tf.minimum(x_adv-x_adv + 1, x_adv + bbox_semi_side)

    ybar = model(x_adv)
    pred = tf.argmax(ybar, axis=1)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2

    #Add a condition to stop when all the labels are flipped! Done!
    def _cond(x_adv, all_flipped):
        return tf.not_equal(True, all_flipped)

    def adjust_size(step_size,dy_dx):
        dy_dx = tf.reshape(dy_dx,[-1,32*32*3]) 
        mag = tf.norm(dy_dx,axis=1, keep_dims=True)
        ans = tf.ones(tf.shape(dy_dx))/(mag)
        ans = ans*(step_size)
        dy_dx = ans*dy_dx
        dy_dx = tf.reshape(dy_dx,[-1,32,32,3])
        return ans


    def _body(x_adv, all_flipped):
        ybar_new, logits = model(x_adv, logits=True)
        pred_new = tf.argmax(ybar_new, axis=1)
        # print (pred.get_shape())
        loss = loss_fn(labels=target, logits=logits)
        not_flipped = tf.equal(pred,pred_new)
        dy_dx, = tf.gradients(loss, x_adv)
        zeroes = tf.zeros(tf.shape(dy_dx), tf.float32)
        ones = tf.ones(tf.shape(dy_dx), tf.float32)
        mask = tf.where(not_flipped, ones, zeroes)
        # print('hi1')
        dy_dx = tf.reshape(dy_dx,[-1,32*32*3])
        mag = tf.norm(dy_dx,axis=1, keepdims=True)
        ans = tf.cond(tf.equal(tf.reduce_sum(mag),0.0),lambda: tf.ones(tf.shape(dy_dx)), lambda: tf.ones(tf.shape(dy_dx))/(mag))
        mask = tf.cond(tf.equal(tf.reduce_sum(mag),0.0),lambda: zeroes, lambda: mask)
        ans = ans*(step_size)
        dy_dx = ans*dy_dx
        dy_dx = tf.reshape(dy_dx,[-1,32,32,3])
        dy_dx = mask*dy_dx
        # print('hi2')
        x_adv = tf.stop_gradient(x_adv + (dy_dx))
        x_adv = tf.clip_by_value(x_adv, x_adv_llimit, x_adv_ulimit)
        all_flipped = tf.equal(tf.reduce_sum(mask), 0.0)
        # print (all_flipped.get_shape())
        return x_adv, all_flipped

    x_adv, all_flipped = tf.while_loop(_cond, _body, (x_adv, False), back_prop=False,
                             name='fgsm')
    return x_adv, all_flipped
