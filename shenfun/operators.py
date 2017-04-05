import numpy as np
from .arguments import Expr

# operators
def div(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    integrals = test.integrals()

    assert integrals.shape[0] == len(space)  # vector
    if len(space) == 1:      # 1D
        v = integrals.copy()
        v += 1
    else:
        ndim = len(space)
        v = integrals.copy()
        for i, s in enumerate(v):
            s[..., i] += 1
        v = v.reshape((1, np.prod(v.shape[:-1]), ndim))
    return Expr(space, v, test.argument())

def grad(test):
    assert isinstance(test, Expr)
    space = test.function_space()
    integrals = test.integrals()

    ndim = len(space)
    assert ndim == integrals.shape[-1]
    assert integrals.shape[0] == 1       # can only take gradient of scalar (for now)
    ss = list(integrals.shape)
    ss[0] *= ndim
    v = np.broadcast_to(integrals, ss).copy()  # vector
    for i, s in enumerate(v):
        s[..., i%ndim] += 1
    return Expr(space, v, test.argument())

def Dx(test, x, k):
    assert isinstance(test, Expr)
    space = test.function_space()
    integrals = test.integrals()

    v = integrals.copy()
    for t in v:
        t[x] = k
    return Expr(space, v, test.argument())

#def Laplace(test):
    #ndim = len(test[0])
    #s = 2*np.identity(ndim, dtype=int)[np.newaxis, :]
    #return (test[0], s)

#def BiharmonicOperator(test):
    #ndim = len(test[0])
    #if ndim == 1:
        #s = np.array([[4]], dtype=np.int)
    #elif ndim == 2:
        #s = np.array([[[4, 0],[0, 4], [2, 2], [2, 2]]], dtype=np.int)
    #elif ndim == 3:
        #s = np.array([[[4, 0, 0],
                      #[0, 4, 0],
                      #[0, 0, 4],
                      #[2, 2, 0],
                      #[2, 2, 0],
                      #[2, 0, 2],
                      #[2, 0, 2],
                      #[0, 2, 2],
                      #[0, 2, 2]]], dtype=np.int)
    #return (test[0], s)
