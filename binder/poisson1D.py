from sympy import symbols, sin, lambdify
from shenfun import *

# Use sympy to compute manufactured solution
x = symbols("x")
ue = sin(4*np.pi*x)*(1-x**2)
fe = ue.diff(x, 2)

SD = Basis(32, 'Chebyshev', bc=(0, 0))
u = TrialFunction(SD)
v = TestFunction(SD)

# Assemble left and right hand
fj = Array(SD, buffer=fe)
f_hat = inner(v, Array(SD, buffer=fe))
A = inner(v, div(grad(u)))

# Solve
u_hat = A/f_hat
uj = u_hat.backward()
print('Error = \n', uj-lambdify(x, ue)(SD.mesh()))
