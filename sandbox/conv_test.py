from numpy import log
from matplotlib import pyplot as plt
import os
os.chdir('../demo')
from poisson1D import main

N = range(10, 200, 20)
error = {}
families = ('legendre', 'chebyshev', 'chebyshevu', 'ultraspherical')
for fam in families:
    error[fam] = []
    for i in range(len(N)):
        Error = main(N[i], fam, 0)
        error[fam].append(Error)
        if i == 0:
            print("Error          hmin           r       ")
            print("%2.8e %2.8e %2.8f"%(error[fam][-1], 1./N[i], 0))
        if i > 0:
            print("%2.8e %2.8e %2.8f"%(error[fam][-1], 1./N[i], log(error[fam][-1]/error[fam][-2])/log(N[i-1]/N[i])))

plt.figure(figsize=(6, 4))
for basis, col in zip(families, 'rbgy'):
    plt.semilogy(N, error[basis], col, linewidth=2)
plt.title('Convergence of Poisson solvers 1D')
#plt.title('Convergence of Helmholtz solvers 1D')
#plt.title('Convergence of Biharmonic solvers 1D')
plt.xlabel('Number of dofs')
plt.ylabel('Error norm')
plt.legend(families)
#plt.text(35, 1e-3, r'With manufactured solution')
#plt.text(40, 5e-5, r'$u=\sin(4\pi x)(1-x^2)$')
plt.savefig('poisson1D_errornorm.png')
plt.show()
