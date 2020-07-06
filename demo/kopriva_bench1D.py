import numpy as np
from matplotlib import pyplot as plt
from shenfun import inner, div, grad, Function, Array, TestFunction, FunctionSpace


def main(N, dt=0.005, end_time=2, dealias_initial=True, plot_result=False):
    SD = FunctionSpace(N, 'F', dtype='D')
    X = SD.points_and_weights()[0]
    v = TestFunction(SD)
    U = Array(SD)
    dU = Function(SD)
    U_hat = Function(SD)
    U_hat0 = Function(SD)
    U_hat1 = Function(SD)
    w0 = Function(SD)
    a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
    b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter

    nu = 0.2
    k = SD.wavenumbers().astype(float)

    # initialize
    U[:] = 3./(5.-4.*np.cos(X))
    if not dealias_initial:
        U_hat = SD.forward(U, U_hat)
    else:
        U_hat[:] = 2**(-abs(k))

    def compute_rhs(rhs, u_hat, w0):
        rhs.fill(0)
        w0.fill(0)
        rhs = inner(v, nu*div(grad(u_hat)), output_array=rhs)
        rhs -= inner(v, grad(u_hat), output_array=w0)
        return rhs

    # Integrate using a 4th order Rung-Kutta method
    t = 0.0
    tstep = 0
    if plot_result is True:
        im = plt.figure()
        ca = im.gca()
        ca.plot(X, U.real, 'b')
        plt.draw()
        plt.pause(1e-6)
    while t < end_time-1e-8:
        t += dt
        tstep += 1
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            dU = compute_rhs(dU, U_hat, w0)
            if rk < 3:
                U_hat[:] = U_hat0 + b[rk]*dt*dU
            U_hat1 += a[rk]*dt*dU
        U_hat[:] = U_hat1

        if tstep % (200) == 0 and plot_result is True:
            ca.plot(X, U.real)
            plt.pause(1e-6)
            #plt.savefig('Ginzburg_Landau_pad_{}_real_{}.png'.format(N[0], int(np.round(t))))

    U = SD.backward(U_hat, U)
    Ue = np.zeros_like(U)
    for k in range(-100, 101):
        Ue += 2**(-abs(k))*np.exp(1j*k*(X-t) - nu*k**2*t)

    err = np.sqrt(2*np.pi/N*np.linalg.norm(U-Ue.real)**2)
    return 1./N, err

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        N = eval(sys.argv[-1])
        print(main(N, plot_result=True, dealias_initial=False))
        plt.show()

    else:
        err = []
        N = range(4, 31, 2)
        for n in N:
            err.append(main(n, dealias_initial=True))

        err = np.array(err)
        for i in range(1, err.shape[0]):
            print("%2.6f %2.4e %2.4e" %(err[i, 0], err[i, 1], np.log(err[i, 1]/err[i-1, 1]) / np.log(err[i, 0]/err[i-1, 0])))

        plt.figure()

        leg = ["N = {}".format(i) for i in N]
        leg.append("Exact")
        plt.legend(leg, loc='lower left')

        plt.plot(1./err[:, 0], np.log10(err[:, 1]))
        plt.xlabel(r'N')
        plt.ylabel('L2 error norm')
        plt.show()
