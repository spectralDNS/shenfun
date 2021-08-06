import subprocess
from numpy import log, array
from matplotlib import pyplot as plt
import os

os.chdir('../demo')
N = range(10, 1000, 50)
error = {}
for basis in ('legendre', 'chebyshev'):
    error[basis] = []
    for i in range(len(N)):
        output = subprocess.check_output("python -O dirichlet_poisson1D.py {} {}".format(N[i], basis), shell=True)
        #output = subprocess.check_output("python dirichlet_Helmholtz1D.py {} {}".format(N[i], basis), shell=True)
        #output = subprocess.check_output("python biharmonic1D.py {} {}".format(N[i], basis), shell=True)
        exec(output) # Error is printed as "Error=%2.16e"%(np.linalg.norm(uj-ua))
        error[basis].append(Error)
        if i == 0:
            print("Error          hmin           r       ")
            print("%2.8e %2.8e %2.8f"%(error[basis][-1], 1./N[i], 0))
        if i > 0:
            print("%2.8e %2.8e %2.8f"%(error[basis][-1], 1./N[i], log(error[basis][-1]/error[basis][-2])/log(N[i-1]/N[i])))

plt.figure(figsize=(6, 4))
for basis, col in zip(('legendre', 'chebyshev'), ('r', 'b')):
    plt.semilogy(N, error[basis], col, linewidth=2)
plt.title('Convergence of Poisson solvers 1D')
#plt.title('Convergence of Helmholtz solvers 1D')
#plt.title('Convergence of Biharmonic solvers 1D')
plt.xlabel('Number of dofs')
plt.ylabel('Error norm')
plt.legend(('Legendre', 'Chebyshev'))
#plt.text(35, 1e-3, r'With manufactured solution')
#plt.text(40, 5e-5, r'$u=\sin(4\pi x)(1-x^2)$')
plt.savefig('poisson1D_errornorm.png')
plt.show()
