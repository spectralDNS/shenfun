r"""
Solve diffusion-convection equation in 1D with possibly inhomogeneous Dirichlet
bcs

.. math::

    \nabla^2 u + \alpha \nabla u &= f, \\
    u(-1) &= b \\
    u(1) &= a

The equation to solve for a Legendre basis is

.. math::

     (\nabla u, \nabla v) - \alpha (\nabla u, v) = -(f, v)

whereas for Chebyshev we solve

.. math::

     (\nabla^2 u, v) + \alpha (\nabla u, v) = (f, v)

Note that this example is complicated by the fact that the boundary conditions
are inhomogeneous. In such a case we use basis

.. math::

    \psi_k &= P_k - P_{k+2} \quad \forall k \in (0, 1, \ldots, N-3) \\
    \psi_{N-2} &= 0.5(P_0 + P_1) \\
    \psi_{N-1} &= 0.5(P_0 - P_1)

where :math:`P_k` is either Chebyshev or Legendre polynomial of order k.

With given expansion we search for a function

.. math::

    u(x) = \sum_{k=0}^{N-1} \hat{u}_k \psi_k

and due to boundary conditions we can immediately close two of the
degrees of freedom, :math:`\hat{u}_{N-2}=a` and :math:`\hat{u}_{N-1}=b`.
These two degrees of freedom must be accounted for in the variational form
that becomes for Chebyshev

.. math::

    {\sum_{k=0}^{N-3}(\psi_j, \nabla^2 \psi_k)  + \alpha \sum_{k=0}^{N-3}(\psi_j, \nabla \psi_k)}\hat{u}_k = \tilde{f}_j - \alpha \sum_{k=N-2}^{N-1}(\psi_0, \nabla \psi_{k})\hat{u}_{k}

solved for all :math:`j=0, 1, \ldots, N-3`. Note the last term on the right hand
side only affects the equation with :math:`j=0`.

"""
import sys
from sympy import symbols, sin, lambdify
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, Basis

assert len(sys.argv) == 3, 'Call with two command-line arguments'
assert sys.argv[-1] in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)

# Get family from args
family = sys.argv[-1].lower()

# Use sympy to compute a rhs, given an analytical solution
domain = (-1., 1.)
alpha = 1.
a = -1.
b = 1.
x = symbols("x")
ue = sin(4*np.pi*x)*(x+domain[0])*(x+domain[1]) + a*(x-domain[0])/2. + b*(domain[1] - x)/2.
fe = ue.diff(x, 2) + alpha*ue.diff(x, 1)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy')
fl = lambdify(x, fe, 'numpy')

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, bc=(a, b), domain=domain)
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fl(X))

# Compute right hand side of Poisson equation
f_hat = Function(SD)
f_hat = inner(v, fj, output_array=f_hat)
if family == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if family == 'chebyshev':
    A = inner(v, div(grad(u))) + alpha*inner(v, grad(u))
else:
    A = inner(grad(v), grad(u)) - alpha*inner(v, grad(u))

if family == 'chebyshev':
    f_hat[0] -= 0.5*np.pi*alpha*a
    f_hat[0] += 0.5*np.pi*alpha*b
elif family == 'legendre':
    f_hat[0] += alpha*a
    f_hat[0] -= alpha*b

f_hat[:-2] = A.solve(f_hat[:-2])
f_hat[-2] = a
f_hat[-1] = b
uj = SD.backward(f_hat)

# Compare with analytical solution
ua = ul(X)
print("Error=%2.16e" %(np.linalg.norm(uj-ua)))
assert np.allclose(uj, ua)

point = np.array([0.1, 0.2])
p = SD.eval(point, f_hat)
assert np.allclose(p, ul(point))
