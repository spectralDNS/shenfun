r"""
Solve Ginzburg-Landau equation on (-50, 50)x(-50, 50) with periodic bcs

    u_t = div(grad(u)) + u - (1+1.5i)*u*|u|**2         (1)

Use Fourier basis V and find u in VxV such that

   (v, u_t) = (v, div(grad(u))+ u) - (v, (1+1.5i)*u*|u|**2)     for all v in VxV


"""
from sympy import symbols, exp
import matplotlib.pyplot as plt
from mpi4py_fft import generate_xdmf, fftw
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    TensorProductSpace, Array, Function, ETDRK4, HDF5File, FunctionSpace, comm

# Use sympy to set up initial condition
x, y = symbols("x,y", real=True)
#ue = (1j*x + y)*exp(-0.03*(x**2+y**2))
ue = (x + y)*exp(-0.03*(x**2+y**2))

# Size of discretization
N = (129, 129)

K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(-50, 50))
K1 = FunctionSpace(N[1], 'F', dtype='D', domain=(-50, 50))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})

Tp = T.get_dealiased((1.5, 1.5))
u = TrialFunction(T)
v = TestFunction(T)

# Try to import wisdom. Note that wisdom must be imported after creating the Bases (that initializes the wisdom somehow?)
try:
    fftw.import_wisdom('GL.wisdom')
    print('Importing wisdom')
except:
    print('No wisdom imported')

# Turn on padding by commenting:
#Tp = T

X = T.local_mesh(True)
U = Array(T, buffer=ue)
Up = Array(Tp)
U_hat = Function(T)

#initialize
U_hat = T.forward(U, U_hat)

def LinearRHS(self, **par):
    L = inner(v, div(grad(u))) + inner(v, u)
    return L

def NonlinearRHS(self, u, u_hat, rhs, **par):
    global Up, Tp
    rhs.fill(0)
    Up = Tp.backward(u_hat, Up)
    rhs = Tp.forward(-(1+1.5j)*Up*abs(Up)**2, rhs)
    return rhs

plt.figure()
image = plt.contourf(X[0], X[1], U.real, 100)
plt.draw()
plt.pause(1e-6)
count = 0
def update(self, u, u_hat, t, tstep, plot_tstep, write_tstep, file, **params):
    global count
    if tstep % plot_tstep == 0 and plot_tstep > 0:
        u = u_hat.backward(u)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], u.real, 100)
        plt.pause(1e-6)
        count += 1
        #plt.savefig('Ginzburg_Landau_{}_{}.png'.format(N[0], count))
    if tstep % write_tstep[0] == 0:
        u = u_hat.backward(u)
        file.write(tstep, write_tstep[1])

if __name__ == '__main__':
    file0 = HDF5File("Ginzburg_Landau_{}.h5".format(N[0]), mode='w')
    par = {'plot_tstep': 100,
           'write_tstep': (50, {'u': [U.real]}),
           'file': file0}
    t = 0.0
    dt = 0.01
    end_time = 100
    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time))
    if comm.Get_rank() == 0:
        generate_xdmf("Ginzburg_Landau_{}.h5".format(N[0]))
    fftw.export_wisdom('GL.wisdom')
