import numpy as np
from matplotlib import pyplot as plt
from shenfun import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import PolyCollection

N = 256
T = Basis(N, 'F', dtype='d', plan=True)
#Tp = T
Tp = Basis(N, 'F', dtype='d', plan=True, padding_factor=1.5)
x = T.points_and_weights(N)[0]
u = TrialFunction(T)
v = TestFunction(T)
k = T.wavenumbers(N, scaled=True, eliminate_highest_freq=True)

u_ = Function(T, False)
Up = Function(Tp, False)
u_hat = Function(T)

def LinearRHS(**params):
    return -inner(Dx(u, 0, 3), v) / (2*np.pi)

def NonlinearRHS(u, u_hat, rhs, **params):
    rhs.fill(0)
    Up[:] = Tp.backward(u_hat, Up)
    #return inner(grad(-0.5*Up**2), v) / (2*np.pi)
    rhs = Tp.forward(-0.5*Up**2, rhs)
    return rhs*1j*k

# initialize
A = 25.
B = 16.
u_[:] = 3*A**2/np.cosh(0.5*A*(x-np.pi+2))**2 + 3*B**2/np.cosh(0.5*B*(x-np.pi+1))**2

u_hat = T.forward(u_, u_hat)
data = []
tdata = []
plt.figure()

def update(u, u_hat, t, tstep, plot_step, **params):
    if tstep % plot_step == 0 and plot_step > 0:
        u = T.backward(u_hat, u)
        plt.plot(x, u)
        plt.draw()
        plt.pause(1e-6)
        data.append(u.copy())

dt = 0.01/N**2
end_time = 0.006
par = {'plot_step': int(end_time/25/dt)}

integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
integrator.setup(dt)
u_hat = integrator.solve(u_, u_hat, dt, (0, end_time))

t = end_time
s = []
for d in data:
    s.append(np.vstack((x, d)).T)

N = len(data)
tdata = np.linspace(0, end_time, N)
ddata = np.array(data)

fig = plt.figure(figsize=(8,3))
ax = axes3d.Axes3D(fig)
X, Y = np.meshgrid(x, tdata)
ax.plot_wireframe(X, Y, ddata, cstride=1000)
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(0, t)
ax.set_zlim(0, 2000)
ax.view_init(65, -105)
ax.set_zticks([0, 2000])
ax.grid()


fig2 = plt.figure(figsize=(8,3))
ax2 = fig2.gca(projection='3d')
poly = PolyCollection(s, facecolors=(1,1,1,1), edgecolors='b')
ax2.add_collection3d(poly, zs=tdata, zdir='y')
ax2.set_xlim3d(0, 2*np.pi)
ax2.set_ylim3d(0, t)
ax2.set_zlim3d(0, 2000)
ax2.view_init(65, -105)
ax2.set_zticks([0, 2000])
ax2.grid()

fig3 = plt.figure(figsize=(8,3))
ax3 = fig3.gca(projection='3d')
X, Y = np.meshgrid(x, tdata)
ax3.plot_surface(X, Y, ddata, cstride=1000, rstride=1, color='w')
ax3.set_xlim(0, 2*np.pi)
ax3.set_ylim(0, t)
ax3.set_zlim(0, 2000)
ax3.view_init(65, -105)
ax3.set_zticks([0, 2000])
ax3.grid()

fig4 = plt.figure(figsize=(8,3))
ax4 = fig4.gca(projection='3d')
for i in range(len(tdata)):
    ax4.plot(x, ddata[i], tdata[i])
ax4.view_init(65, -105)
ax4.set_zticks([0, 2000])
ax4.grid()

fig5 = plt.figure(facecolor='k')
ax5 = fig5.add_subplot(111, axisbg='k')
N = len(tdata)
for i in range(N):
    offset = (N-i-1)*200
    ax5.plot(x, ddata[N-i-1]+offset, 'w', lw=2, zorder=(i+1)*2)
    ax5.fill_between(x, ddata[N-i-1]+offset, offset, facecolor='k', lw=0, zorder=(i+1)*2-1)
fig5.savefig('KdV.png')
#plt.show()
