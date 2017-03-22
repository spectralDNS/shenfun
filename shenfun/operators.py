import numpy as np

# operators
def div(test):
    assert test[1].shape[0] == len(test[0])  # vector
    if len(test[0]) == 1:      # 1D
        v = test[1].copy()
        v += 1
    else:
        ndim = len(test[0])
        v = test[1].copy()
        for i, s in enumerate(v):
            s[..., i] += 1
        v = v.reshape((1, np.prod(v.shape[:-1]), ndim))
    return (test[0], v)

def grad(test):
    v = test[1]
    ndim = len(test[0])
    assert v.shape[0] == 1       # scalar
    ss = list(v.shape)
    ss[0] *= ndim
    vv = np.broadcast_to(v, ss).copy()  # vector
    for i, s in enumerate(vv):
        s[..., i%ndim] += 1
    return (test[0], vv)

def Dx(test, x, k):
    v = test[1].copy()
    for t in v:
        t[x] = k
    return (test[0], v)

def Laplace(test):
    ndim = len(test[0])
    s = 2*np.identity(ndim, dtype=int)[np.newaxis, :]
    return (test[0], s)

def BiharmonicOperator(test):
    ndim = len(test[0])
    if ndim == 1:
        s = np.array([[4]], dtype=np.int)
    elif ndim == 2:
        s = np.array([[[4, 0],[0, 4], [2, 2], [2, 2]]], dtype=np.int)
    elif ndim == 3:
        s = np.array([[[4, 0, 0],
                      [0, 4, 0],
                      [0, 0, 4],
                      [2, 2, 0],
                      [2, 2, 0],
                      [2, 0, 2],
                      [2, 0, 2],
                      [0, 2, 2],
                      [0, 2, 2]]], dtype=np.int)
    return (test[0], s)
