import numpy as np
sd = 'shape_dict'

def save_obj(obj, name):
    np.save('data/misc/'+name +'.npy', obj)


def load_obj(name):
    return np.load('data/misc/'+name +'.npy').item()


def save_as_txt(label, ar):
    d = load_obj(sd)
    d[label] = ar.shape
    save_obj(d, sd)
    X = ar.reshape((ar.shape[0], -1))
    np.savetxt(label, X)


def load_from_txt(label):
    d = load_obj(sd)
    X = np.loadtxt(label)
    return X.reshape(d[label])