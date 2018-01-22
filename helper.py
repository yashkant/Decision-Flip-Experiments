import numpy as np
import _pickle as pickle

img_rows = 32
img_cols = 32
img_chas = 3
input_shape = (img_rows, img_cols, img_chas)
n_classes = 10

layer_shapes = {
    '0': [32, 32, 3],
    '1': [16, 16, 32],
    '2': [8, 8, 64],
    '3': [1024],
    '4': [128],
    '5': [10]
}


def load_CIFAR10(ROOT, os):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def find_l2(X_test, X_adv):
    a = X_test.reshape(-1, 32 * 32 * 3)
    b = X_adv.reshape(-1, 32 * 32 * 3)
    l2_unsquared = np.sum(np.square(a - b), axis=1)
    return l2_unsquared


def find_l2_batch(X_test, X_adv):
    ans = np.zeros([X_test.shape[0], n_classes], dtype=np.float32)
    for i in range(X_test.shape[0]):
        for j in range(n_classes):
            ans[i][j] = find_l2(X_test[i], X_adv[i][j])
    return ans


def remove_zeroes(X):
    indices = np.where(X == 0)[0]
    return np.delete(X, indices)


def get_class(X, Y, cls, Z=None):
    p = np.argmax(Y, axis=1)
    indices = np.where(p == cls)
    if Z is not None:
        return X[indices], Y[indices], Z[indices]
    else:
        return X[indices], Y[indices]


def make_label(i, m, e, n, r):
    if (r == False):
        return i + "_m" + str(m) + "_e" + str(e) + "_n" + str(n)
    else:
        base = i + "_m" + str(m) + "_e" + str(e) + "_n" + str(n) + "_r"
        lrn = base + "normal"
        return base, lrn


def get_flipped_class(X_adv, cls):
    return X_adv[:, cls]


def get_flip_path(l):
    return 'data/cifar/' + l + '.txt'


def get_misc_path(l):
    return 'data/misc/' + l + '.txt'


def count_clear(l2_test, l2_train, l2_random, l2_random_normal):
    nz_test = np.count_nonzero(l2_test)
    nz_train = np.count_nonzero(l2_train)
    nz_random = np.count_nonzero(l2_random)
    nz_random_normal = np.count_nonzero(l2_random_normal)

    print('\n test: ' + str(nz_test))
    print('train: ' + str(nz_train))
    print('random: ' + str(nz_random))
    print('random normal: ' + str(nz_random_normal))

    l2_test = remove_zeroes(l2_test)
    l2_random = remove_zeroes(l2_random)
    l2_random_normal = remove_zeroes(l2_random_normal)
    l2_train = remove_zeroes(l2_train)

    min_no = min(nz_test, nz_train)
    l2_train = np.sqrt(l2_train[:min_no])
    l2_test = np.sqrt(l2_test[:min_no])
    l2_random = np.sqrt(l2_random[:min_no])
    l2_random_normal = np.sqrt(l2_random_normal[:min_no])

    return l2_test, l2_train, l2_random, l2_random_normal


# one hot encoding
def _to_categorical(x, n_classes):
    x = np.array(x, dtype=int).ravel()
    n = x.shape[0]
    ret = np.zeros((n, n_classes))
    ret[np.arange(n), x] = 1
    return ret


# m2 is the grouped flipping
# m1 is the single flipping
# This method returns the distance of each predictions from repective test points calculated by m1 and m2 resp.
def find_m1_m2(X_test, X_adv_one, X_adv_test):
    dist_adv_m1 = find_l2(X_test, X_adv_one)
    b = find_l2_batch(X_test, X_adv_test)
    dist_adv_m2 = np.partition(b, axis=1, kth=1)[:, 1]
    return np.sqrt(dist_adv_m1), np.sqrt(dist_adv_m2)


# Give this function X_adv_test it gives you the points corresponding to
# each example having min dists and their indices
def give_m2_ans(X_test, X_adv_test, cls=-1):
    if (cls == -1):
        dists = find_l2_batch(X_test, X_adv_test)
        second_min_indices = np.partition(dists, axis=1, kth=1)[:, 1]
        for i in range(X_test.shape[0]):
            second_min_indices[i] = (np.where(second_min_indices[i] == dists[i])[0][0])
        ans = np.empty([X_adv_test.shape[0], img_rows, img_cols, img_chas])
        for i in range(ans.shape[0]):
            ans[i] = X_adv_test[i][second_min_indices[i].astype(int)]
        return second_min_indices, ans
    else:
        return 0, get_flipped_class(X_adv_test, cls)


def load_data(os):
    print('\nLoading CIFAR10')
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir, os)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    X_train = X_train.reshape(-1, img_rows, img_cols, img_chas)
    X_test = X_test.reshape(-1, img_rows, img_cols, img_chas)

    y_train = _to_categorical(y_train, n_classes)
    y_test = _to_categorical(y_test, n_classes)

    print('\nLoading pre-Shuffled training data')
    ind = np.loadtxt(get_misc_path('ind'), dtype=int)
    X_train, y_train = X_train[ind], y_train[ind]

    # split training/validation dataset
    validation_split = 0.1
    n_train = int(X_train.shape[0] * (1 - validation_split))
    X_valid = X_train[n_train:]
    X_train = X_train[:n_train]
    y_valid = y_train[n_train:]
    y_train = y_train[:n_train]

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def remove_zeroes(X):
    indices = np.where(X == 0)[0]
    return np.delete(X, indices)


def get_class(X, Y, Z, cls):
    p = np.argmax(Y, axis=1)
    indices = np.where(p == cls)
    return X[indices], Y[indices], Z[indices]


def make_label(i, m, e, n, r):
    if (r == False):
        return i + "_m" + str(m) + "_e" + str(e) + "_n" + str(n)
    else:
        base = i + "_m" + str(m) + "_e" + str(e) + "_n" + str(n) + "_r"
        lrn = base + "normal"
        return base, lrn


def get_flipped_class(X_adv, cls):
    return X_adv[:, cls]


def get_l2_at_layer(X, X_flip, sess, env, layer=-1):
    X = sess.run(env.layer_out, feed_dict={
        env.x: X, env.training: False, env.lyr: layer})

    X_flip = sess.run(env.layer_out, feed_dict={
        env.x: X_flip, env.training: False, env.lyr: layer})

    # print(X.shape)

    if layer == -1:
        dim2 = np.prod(layer_shapes['0'])
    else:
        dim2 = np.prod(layer_shapes[str(layer)])
    # print(dim2)
    a = X.reshape(-1, dim2)
    b = X_flip.reshape(-1, dim2)

    l2_unsquared = np.sum(np.square(a - b), axis=1)

    return l2_unsquared


def find_l2_batch(X_test, X_adv):
    ans = np.zeros([X_test.shape[0], n_classes], dtype=np.float32)
    for i in range(X_test.shape[0]):
        for j in range(n_classes):
            ans[i][j] = find_l2(X_test[i], X_adv[i][j])
    return ans


# m2 is the grouped flipping
# m1 is the single flipping
# This method returns the distance of each predictions from repective test points calculated by m1 and m2 resp.
def find_m1_m2(X_test, X_adv_one, X_adv_test):
    dist_adv_m1 = find_l2(X_test, X_adv_one)
    b = find_l2_batch(X_test, X_adv_test)
    dist_adv_m2 = np.partition(b, axis=1, kth=1)[:, 1]
    return np.sqrt(dist_adv_m1), np.sqrt(dist_adv_m2)


# Give this function X_adv_test it gives you the points corresponding to
# each example having min dists and their indices
def give_m2_ans(X_test, X_adv_test, cls=-1):
    if (cls == -1):
        dists = find_l2_batch(X_test, X_adv_test)
        second_min_indices = np.partition(dists, axis=1, kth=1)[:, 1]
        for i in range(X_test.shape[0]):
            second_min_indices[i] = (np.where(second_min_indices[i] == dists[i])[0][0])
        ans = np.empty([X_adv_test.shape[0], img_rows, img_cols, img_chas])
        for i in range(ans.shape[0]):
            ans[i] = X_adv_test[i][second_min_indices[i].astype(int)]
        return second_min_indices, ans
    else:
        return 0, get_flipped_class(X_adv_test, cls)


# Redundant Old Method!
def create_adv(X, Y, label):
    print('\nCrafting adversarial')
    n_sample = X.shape[0]
    batch_size = 1
    n_batch = int(np.ceil(n_sample / batch_size))
    n_epoch = 20
    X_adv = np.empty_like(X)
    for ind in range(n_batch):
        print(' batch {0}/{1}'.format(ind + 1, n_batch), end='\r')
        start = ind * batch_size
        end = min(n_sample, start + batch_size)
        tmp, all_flipped = sess.run([env.x_adv, env.all_flipped], feed_dict={env.x: X[start:end],
                                                                             env.y: Y[start:end],
                                                                             env.training: False})
        X_adv[start:end] = tmp
    print('\nSaving adversarial')
    os.makedirs('data', exist_ok=True)
    save_as_txt(get_flip_path(label), X_adv)
    return X_adv
