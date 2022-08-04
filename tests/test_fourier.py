import pytest
from shenfun import FunctionSpace, TensorProductSpace, Function, np, comm, fourier


@pytest.mark.parametrize('N', ((12,)*3, (13,)*3))
def test_energy_fourier(N):
    B0 = FunctionSpace(N[0], 'F', dtype='D')
    B1 = FunctionSpace(N[1], 'F', dtype='D')
    B2 = FunctionSpace(N[2], 'F', dtype='d')
    for bases, axes in zip(((B0, B1, B2), (B0, B2, B1)),
                           ((0, 1, 2), (2, 0, 1))):
        T = TensorProductSpace(comm, bases, axes=axes)
        u_hat = Function(T)
        u_hat[:] = np.random.random(u_hat.shape) + 1j*np.random.random(u_hat.shape)
        u = u_hat.backward()
        u_hat = u.forward(u_hat)
        u = u_hat.backward(u)
        e0 = comm.allreduce(np.sum(u.v*u.v)/np.prod(N))
        e1 = fourier.energy_fourier(u_hat, T)
        assert abs(e0-e1) < 1e-10
        T.destroy()

if __name__ == '__main__':
    test_energy_fourier((12, 12, 12))
