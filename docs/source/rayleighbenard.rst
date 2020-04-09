.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

Demo - Rayleigh Benard
%%%%%%%%%%%%%%%%%%%%%%

:Authors: Mikael Mortensen (mikaem at math.uio.no)
:Date: Apr 9, 2020

*Summary.* Rayleigh-Benard convection arise
due to temperature gradients in a fluid. The governing equations are
Navier-Stokes coupled (through buoyancy) with an additional temperature
equation derived from the first law of thermodynamics, using a linear
correlation between density and temperature.

This is a demonstration of how the Python module `shenfun <https://github.com/spectralDNS/shenfun>`__ can be used to solve for
these Rayleigh-Benard cells in a 2D channel with two walls of
different temperature in one direction, and periodicity in the other direction.
The solver described runs with MPI
without any further considerations required from the user.
Note that there is a more physically realistic 3D solver implemented within
`the spectralDNS project <https://github.com/spectralDNS/spectralDNS/blob/master/spectralDNS/solvers/KMMRK3_RB.py>`__.
To allow for some simple optimizations, the solver described in this demo has been implemented in a class in the
`RayleighBenardRk3.py <https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenardRK3.py>`__
module in the demo folder of shenfun. Below are two example solutions, where the first (movie)
has been run at a very high Rayleigh number (*Ra*), and the lower image with a low *Ra* (laminar).

.. _fig:RB:

.. figure:: https://raw.githack.com/spectralDNS/spectralutilities/master/movies/RB_100x256_100k_fire.png
   :width: 800

   Temperature fluctuations in the Rayleigh Benard flow. The top and bottom walls are kept at different temperatures and this sets up the Rayleigh-Benard convection. The simulation is run at *Ra* =100,000, *Pr* =0.7 with 100 and 256 quadrature points in *x* and *y*-directions, respectively

.. _fig:RB_lam:

.. figure:: https://raw.githack.com/spectralDNS/spectralutilities/master/figures/RB_40x128_100_fire.png
   :width: 800

   Convection cells for a laminar flow. The simulation is run at *Ra* =100, *Pr* =0.7 with 40 and 128 quadrature points in *x* and *y*-directions, respectively

.. _demo:rayleighbenard:

Model problem
=============

The governing equations solved in domain :math:`\Omega=[-1, 1]\times [0, 2\pi]` are

.. math::
   :label: eq:momentum

        
            \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = - \nabla p + \sqrt{\frac{Pr}{Ra}} \nabla^2 \mathbf{u}  + T \mathbf{i}, 
        

.. math::
   :label: eq:T

          
            \frac{\partial T}{\partial t} +\mathbf{u} \cdot \nabla T = \frac{1}{\sqrt{RaPr}} \nabla^2 T, 
        

.. math::
   :label: eq:div

          
            \nabla \cdot \mathbf{u} = 0, 
        

where :math:`\mathbf{u}(x, y, t) (= u\mathbf{i} + v\mathbf{j})` is the velocity vector, :math:`p(x, y, t)` is pressure, :math:`T(x, y, t)` is the temperature, and :math:`\mathbf{i}` and
:math:`\mathbf{j}` are the unity vectors for the :math:`x` and :math:`y`-directions, respectively.

The equations are complemented with boundary conditions :math:`\mathbf{u}(\pm 1, y, t) = (0, 0), \mathbf{u}(x, 2 \pi, t) = \mathbf{u}(x, 0, t), T(1, y, t) = 1, T(-1, y, t) =  0, T(x, 2 \pi, t) = T(x, 0, t)`.
Note that these equations have been non-dimensionalized according to :cite:`pandey18`, using dimensionless
Rayleigh number :math:`Ra=g \alpha \Delta T h^3/(\nu \kappa)` and Prandtl number :math:`Pr=\nu/\kappa`. Here
:math:`g \mathbf{i}` is the vector accelleration of gravity, :math:`\Delta T` is the temperature difference between
the top and bottom walls, :math:`h` is the hight of the channel in :math:`x`-direction, :math:`\nu` is the
dynamic viscosity coefficient, :math:`\kappa` is the heat transfer coefficient and :math:`\alpha` is the
thermal expansion coefficient. Note that the
governing equations have been non-dimensionalized using the free-fall velocityscale
:math:`U=\sqrt{g \alpha \Delta T h}`. See :cite:`pandey18` for more details.

The governing equations contain a non-trivial coupling between velocity, pressure and temperature.
This coupling can be simplified by eliminating the pressure from the equation for the wall-normal velocity
component :math:`u`. We accomplish this by taking the Laplace of the momentum equation in wall normal
direction, using the pressure from the divergence of the momentum equation 
:math:`\nabla^2 p = -\nabla \cdot \mathbf{H}+\partial T/\partial x`, where
:math:`\mathbf{H} = (H_x, H_y) = (\mathbf{u} \cdot \nabla) \mathbf{u}`

.. math::
   :label: eq:u

        
            \frac{\partial \nabla^2 {u}}{\partial t} = \frac{\partial^2 H_y}{\partial x \partial y} - \frac{\partial^2 H_x}{\partial y\partial y}  + \sqrt{\frac{Pr}{Ra}} \nabla^4 {u}  + \frac{\partial^2 T}{\partial y^2} . 
        

This equation is solved with :math:`u(\pm 1) = \partial u/\partial x(\pm 1) = 0`, where the latter follows from the
divergence constraint. In summary, we now seem to have the following equations to solve:

.. math::
   :label: eq:u2

        
            \frac{\partial \nabla^2 {u}}{\partial t} = \frac{\partial^2 H_y}{\partial x \partial y} - \frac{\partial^2 H_x}{\partial y\partial y}  + \sqrt{\frac{Pr}{Ra}} \nabla^4 {u}  + \frac{\partial^2 T}{\partial y^2}, 
        

.. math::
   :label: eq:v

          
            \frac{\partial v}{\partial t} + H_y = -  \frac{\partial p}{\partial y} + \sqrt{\frac{Pr}{Ra}} \nabla^2 v, 
        

.. math::
   :label: eq:T2

          
            \frac{\partial T}{\partial t} +\mathbf{u} \cdot \nabla T = \frac{1}{\sqrt{RaPr}} \nabla^2 T, 
        

.. math::
   :label: eq:div2

          
            \nabla \cdot \mathbf{u} = 0 .
        

However, we note that Eqs. :eq:`eq:u2` and :eq:`eq:T2` and :eq:`eq:div2` do not depend on pressure, and,
apparently, on each time step we can solve :eq:`eq:u2` for :math:`u`, then :eq:`eq:div2` for :math:`v` and finally :eq:`eq:T2` for :math:`T`.
So what do we need :eq:`eq:v` for? It appears to have become redundant from the elimination of the
pressure from Eq. :eq:`eq:u2`. It turns out that this is actually almost completely true, but
:eq:`eq:u2`, :eq:`eq:T2` and :eq:`eq:div2` can only provide closure for all but one of the
Fourier coefficients. To see this it is necessary to introduce some discretization and basis functions
that will be used to solve the problem. To this end we use :math:`P_N`, which is the set of all real polynomials
of degree less than or equal to N and introduce the following finite-dimensional approximation spaces

.. math::
   :label: eq:VB

        
          V_N^B(x) = \{v \in P_N | v(\pm 1) = v´(\pm 1) = 0\},  
        

.. math::
   :label: eq:VD

          
          V_N^D(x) = \{v \in P_N | v(\pm 1) = 0\},  
        

.. math::
   :label: eq:VT

          
          V_N^T(x) = \{v \in P_N | v(-1) = 0, v(1) = 1\},  
        

.. math::
   :label: eq:VW

          
          V_N^W(x) = \{v \in P_N\},  
        

.. math::
   :label: eq:VF

          
          V_M^F(y) = \{\exp(\imath l y) | l \in [-M/2, -M/2+1, \ldots M/2-1]\}. 
        
        

Here :math:`\text{dim}(V_N^B) = N-4, \text{dim}(V_N^D) = \text{dim}(V_N^W) = N-2`, :math:`\text{dim}(V_N^T) = N`
and :math:`\text{dim}(V_M^F)=M`. We note that
:math:`V_N^B, V_N^D, V_N^W` and :math:`V_N^T` can be used to approximate :math:`u, v, T` and :math:`p`, respectively, in the :math:`x`-direction.
Also note that for :math:`V_M^F` it is assumed that :math:`M` is an even number.

We can now choose basis functions for the spaces, using Shen's composite bases for either Legendre or
Chebyshev polynomials. For the Fourier space the basis functions are already given. We leave the actual choice
of basis as an implementation option for later. For now we use :math:`\phi^B(x), \phi^D(x), \phi^W` and :math:`\phi^T(x)`
as common notation for basis functions in spaces :math:`V_N^B, V_N^D, V_N^W` and :math:`V_N^T`, respectively.

To get the required approximation spaces for the entire domain we use tensor products of the
one-dimensional spaces in :eq:`eq:VB`-:eq:`eq:VF`

.. math::
   :label: eq:WBF

        
          W_{BF} = V_N^B \otimes V_M^F,   
        

.. math::
   :label: eq:WDF

          
          W_{DF} = V_N^D \otimes V_M^F,   
        

.. math::
   :label: eq:WTF

          
          W_{TF} = V_N^T \otimes V_M^F,   
        

.. math::
   :label: eq:WWF

          
          W_{WF} = V_N^W \otimes V_M^F. 
        

Space :math:`W_{BF}` has 2D tensor product basis functions :math:`\phi_k^B(x) \exp (\imath l y)` and
similar for the others. All in all
we get the following approximations for the unknowns

.. math::
   :label: _auto1

        
            u_N(x, y, t) = \sum_{k \in \boldsymbol{k}_B} \sum_{l \in \boldsymbol{l}} \hat{u}_{kl}(t) \phi_k^B(x) \exp(\imath l y), 
        
        

.. math::
   :label: _auto2

          
            v_N(x, y, t) = \sum_{k \in \boldsymbol{k}_D} \sum_{l \in \boldsymbol{l}} \hat{v}_{kl}(t) \phi_k^D(x) \exp(\imath l y), 
        
        

.. math::
   :label: _auto3

          
            p_N(x, y, t) = \sum_{k \in \boldsymbol{k}_W} \sum_{l \in \boldsymbol{l}} \hat{p}_{kl}(t) \phi_k^W(x) \exp(\imath l y), 
        
        

.. math::
   :label: _auto4

          
            T_N(x, y, t) = \sum_{k \in \boldsymbol{k}_T} \sum_{l \in \boldsymbol{l}} \hat{T}_{kl}(t) \phi_k^T(x) \exp(\imath l y),
        
        

where :math:`\boldsymbol{k}_{x} = \{0, 1, \ldots \text{dim}(V_N^x)-1\}, \, \text{for} \, x\in(B, D, W, T)`
and :math:`\boldsymbol{l} = \{-M/2, -M/2+1, \ldots, M/2-1\}`.
Note that since the problem is defined in real space we will have Hermitian symmetry. This means
that :math:`\hat{u}_{k, l} = \overline{\hat{u}}_{k, -l}`, with an overbar being a complex conjugate,
and similar for :math:`\hat{v}_{kl}, \hat{p}_{kl}` and
:math:`\hat{T}_{kl}`. For this reason we can get away with
solving for only the positive :math:`l`'s, as long as we remember that the sum in the end goes over both positive
and negative :math:`l's`. This is actually automatically taken care of by the FFT provider and is
not much of an additional complexity in the implementation. So from now on :math:`\boldsymbol{l} = \{0, 1, \ldots, M/2\}`.

We can now take a look at why Eq. :eq:`eq:v` is needed. If we first solve :eq:`eq:u2` for
:math:`\hat{u}_{kl}(t), (k, l) \in \boldsymbol{k}_B \times \boldsymbol{l}`, then we can use :eq:`eq:div2` to
solve for :math:`\hat{v}_{kl}(t)`. But here there is a problem. We can see this by creating the variational
form required to solve :eq:`eq:div2` by the spectral Galerkin method. To this end make :math:`v=v_N` in :eq:`eq:div2`
a trial function, use :math:`u=u_N` a known function and take the weighted inner product over the
domain using test function :math:`q \in W_{DF}`

.. math::
   :label: _auto5

        
            \left < \frac{\partial u_N}{\partial x} + \frac{\partial v_N}{\partial y}, q \right > _w = 0.
        
        

Here we are using the inner product notation

.. math::
   :label: _auto6

        
            \left < a, b \right > _w = \int_{-1}^1 \int_0^{2\pi} a \overline{b} dx_wdy_w \left(\approx \sum_{i}\sum_{j} a(x_i, y_j) \overline{b}(x_i, y_j) w(x_i) w(y_j)\right),
        
        

where the exact form of the
weighted scalar product depends on the chosen basis; Legendre has :math:`dx_w=dx`, Chebyshev
:math:`dx_w = dx/\sqrt{1-x^2}` and Fourier :math:`dy_w=dy/2/\pi`. The bases also have associated quadrature weights
:math:`\{w(x_i) \}_{i=0}^{N-1}` and :math:`\{w(y_j)\}_{j=0}^{M-1}` that are used to approximate the integrals.

Inserting now for the known :math:`u_N`, the unknown :math:`v_N`, and :math:`q=\phi_m^D(x) \exp(\imath n y)` the
continuity equation becomes

.. math::
   :label: eq:u4

        
          \int_{-1}^1 \int_{0}^{2\pi} \frac{\partial}{\partial x} \left(\sum_{k \in \boldsymbol{k}_B} \sum_{l \in \boldsymbol{l}} \hat{u}_{kl}(t) \phi_k^B(x) \exp(\imath l y) \right) \phi_m^D(x) \exp(-\imath n y) dx_w dy_w + \\ 
          \int_{-1}^1 \int_{0}^{2\pi} \frac{\partial}{\partial y} \left(\sum_{k \in \boldsymbol{k}_D} \sum_{l \in \boldsymbol{l}} \hat{v}_{kl}(t) \phi_k^D(x) \exp(\imath l y) \right) \phi_m^D(x) \exp(-\imath n y) dx_w dy_w  = 0. 
        

The :math:`x` and :math:`y` domains are separable, so we can rewrite as

.. math::
   :label: _auto7

        
            \sum_{k \in \boldsymbol{k}_B} \sum_{l \in \boldsymbol{l}} \int_{-1}^1 \frac{\partial \phi_k^B(x)}{\partial x}  \phi_m^D(x) dx_w \int_{0}^{2\pi} \exp(\imath l y) \exp(-\imath n y) dy_w \hat{u}_{kl} + \\ 
            \sum_{k \in \boldsymbol{k}_D} \sum_{l \in \boldsymbol{l}} \int_{-1}^1 \phi_k^D(x) \phi_m^D(x) dx_w   \int_{0}^{2\pi} \frac{\partial \exp(\imath l y)}{\partial y} \exp(-\imath n y) dy_w \hat{v}_{kl} = 0.
        
        

Now perform some exact manipulations in the Fourier direction and introduce the
1D inner product notation for the :math:`x`-direction

.. math::
   :label: _auto8

        
            \left(a, b\right)_w = \int_{-1}^1 a(x) b(x) dx_w \left(\approx \sum_{j = 0}^{N-1} a(x_j)b(x_j) w(x_j)\right).
        
        

By also simplifying the notation using summation of repeated indices,
we get the following equation

.. math::
   :label: _auto9

        
           \delta_{ln} \left(\frac{\partial \phi_k^B}{\partial x}, \phi_m^D \right)_w \hat{u}_{kl}
           + \imath l \delta_{ln} \left(\phi_k^D, \phi_m^D \right)_w \hat{v}_{kl}  = 0.
        
        

Now :math:`l` must equal :math:`n` and we can simplify some more

.. math::
   :label: eq:div3

        
           \left(\frac{\partial \phi_k^B}{\partial x}, \phi_m^D \right)_w \hat{u}_{kl}
           + \imath l \left(\phi_k^D, \phi_m^D \right)_w \hat{v}_{kl}  = 0. 
        

We see that this equation can be solved for
:math:`\hat{v}_{kl} \text{ for } (k, l) \in \boldsymbol{k}_D \times [1, 2, \ldots, M/2]`, but try with
:math:`l=0` and you hit division by zero, which obviously is not allowed. And this is the reason
why Eq. :eq:`eq:v` is still needed, to solve for :math:`\hat{v}_{k,0}`! Fortunately,
since :math:`\exp(\imath 0 y) = 1`, the pressure derivative :math:`\frac{\partial p}{\partial y} = 0`,
and as such the pressure is still not required. When used only for
Fourier coefficient 0, Eq. :eq:`eq:v` becomes

.. math::
   :label: eq:vx

        
        \frac{\partial v}{\partial t} + N_y = \sqrt{\frac{Pr}{Ra}} \nabla^2 v. 
        

There is still one more revelation to be made from Eq. :eq:`eq:div3`. When :math:`l=0` we get

.. math::
   :label: _auto10

        
            \left(\frac{\partial \phi_k^B}{\partial x}, \phi_m^D \right)_w \hat{u}_{k,0} = 0,
        
        

and the only way to satisfy this is if :math:`\hat{u}_{k,0}=0` for :math:`k\in\boldsymbol{k}_B`. Bottom line is
that we only need to solve Eq. :eq:`eq:u2` for :math:`l \in \boldsymbol{l}/\{0\}`, whereas we can use
directly :math:`\hat{u}_{k,0}=0 \text{ for } k \in \boldsymbol{k}_B`.

To sum up, with the solution known at :math:`t = t - \Delta t`, we solve

================  ===========================  ===================================================================  
    Equation              For unknown                                      With indices                             
================  ===========================  ===================================================================  
 :eq:`eq:u2`       :math:`\hat{u}_{kl}(t)`      :math:`(k, l) \in \boldsymbol{k}_B \times \boldsymbol{l}/\{0\}`  
:eq:`eq:div2`      :math:`\hat{v}_{kl}(t)`      :math:`(k, l) \in \boldsymbol{k}_D \times \boldsymbol{l}/\{0\}`  
 :eq:`eq:vx`       :math:`\hat{v}_{kl}(t)`              :math:`(k, l) \in \boldsymbol{k}_D \times \{0\}`         
 :eq:`eq:T2`       :math:`\hat{T}_{kl}(t)`         :math:`(k, l) \in \boldsymbol{k}_T \times \boldsymbol{l}`     
================  ===========================  ===================================================================  

Temporal discretization
=======================

The governing equations are integrated in time using a semi-implicit third order Runge Kutta method.
This method applies to any generic equation

.. math::
   :label: eq:genericpsi

        
         \frac{\partial \psi}{\partial t} = \mathcal{N} + \mathcal{L}\psi ,
        

where :math:`\mathcal{N}` and :math:`\mathcal{L}` represents the nonlinear and linear contributions, respectively.
With time discretized as :math:`t_n = n \Delta t, \, n = 0, 1, 2, ...`, the
Runge Kutta method also subdivides each timestep into stages
:math:`t_n^k = t_n + c_k \Delta t, \, k = (0, 1, .., N_s-1)`, where :math:`N_s` is
the number of stages. The third order Runge Kutta method implemented here uses three stages.
On one timestep the generic equation :eq:`eq:genericpsi`
is then integrated from stage :math:`k` to :math:`k+1` according to

.. math::
   :label: _auto11

        
            \psi^{k+1} = \psi^k + a_k \mathcal{N}^k + b_k \mathcal{N}^{k-1} + \frac{a_k+b_k}{2}\mathcal{L}(\psi^{k+1}+\psi^{k}),
        
        

which should be rearranged with the unknowns on the left hand side and the
knowns on the right hand side

.. math::
   :label: eq:rk3stages

        
            \big(1-\frac{a_k+b_k}{2}\mathcal{L}\big)\psi^{k+1} = \big(1 + \frac{a_k+b_k}{2}\mathcal{L}\big)\psi^{k} + a_k \mathcal{N}^k + b_k \mathcal{N}^{k-1}. 
        

For the three-stage third order Runge Kutta method the constants are given as

====================  ====================  ======================  
:math:`a_n/\Delta t`  :math:`b_n/\Delta t`  :math:`c_n / \Delta t`  
====================  ====================  ======================  
        8/15                   0                      0             
        5/12                 −17/60                  8/15           
        3/4                  −5/12                   2/3            
====================  ====================  ======================  

For the spectral Galerkin method used by ``shenfun`` the governing equation
is first put in a weak variational form. This will change the appearence of
Eq. :eq:`eq:rk3stages` slightly. If :math:`\phi` is a test function, :math:`\psi^{k+1}`
the trial function, and :math:`\psi^{k}` a known function, then the variational form
of :eq:`eq:rk3stages` is obtained by multiplying :eq:`eq:rk3stages` by :math:`\phi` and
integrating (with weights) over the domain

.. math::
   :label: eq:rk3stagesvar

        
            \Big < (1-\frac{a_k+b_k}{2}\mathcal{L})\psi^{k+1}, \phi \Big > _w = \Big < (1 + \frac{a_k+b_k}{2}\mathcal{L})\psi^{k}, \phi\Big > _w + \Big < a_k \mathcal{N}^k + b_k \mathcal{N}^{k-1}, \phi \Big > _w. 
        

Equation :eq:`eq:rk3stagesvar` is the variational form implemented by ``shenfun`` for the
time dependent equations.

Shenfun implementation
======================

To get started we need instances of the approximation spaces discussed in
Eqs. :eq:`eq:VB` - :eq:`eq:WWF`. When the spaces are created we also need
to specify the family and the dimension of each space. Here we simply
choose Chebyshev and Fourier with 100 and 256 quadrature points in :math:`x` and
:math:`y`-directions, respectively. We could replace 'Chebyshev' by 'Legendre',
but the former is known to be faster due to the existence of fast transforms.

.. code-block:: python

    from shenfun import *
    
    N, M = 100, 256
    family = 'Chebyshev'
    VB = Basis(N, family, bc='Biharmonic')
    VD = Basis(N, family, bc=(0, 0))
    VW = Basis(N, family)
    VT = Basis(N, family, bc=(0, 1))
    VF = Basis(M, 'F', dtype='d')

And then we create tensor product spaces by combining these bases (like in Eqs. :eq:`eq:WBF`-:eq:`eq:WWF`).

.. code-block:: python

    W_BF = TensorProductSpace(comm, (VB, VF))    # Wall-normal velocity
    W_DF = TensorProductSpace(comm, (VD, VF))    # Streamwise velocity
    W_WF = TensorProductSpace(comm, (VW, VF))    # No bc
    W_TF = TensorProductSpace(comm, (WT, VF))    # Temperature
    BD = MixedTensorProductSpace([W_BF, W_DF])   # Velocity vector
    DD = MixedTensorProductSpace([W_DF, W_DF])   # Convection vector

Here the last two lines create mixed tensor product spaces by the
Cartesian products ``BD = W_BF`` :math:`\times` ``W_DF`` and ``DD = W_DF`` :math:`\times` ``W_DF``.
These mixed space will be used to hold the velocity and convection vectors,
but we will not solve the equations in a coupled manner and continue in the
segregated approach outlined above.

We also need containers for the computed solutions. These are created as

.. code-block:: python

    u_  = Function(BD)     # Velocity vector, two components
    u_1 = Function(BD)     # Velocity vector, previous step
    T_  = Function(W_TF)   # Temperature
    T_1 = Function(W_TF)   # Temperature, previous step
    H_  = Function(DD)     # Convection vector
    H_1 = Function(DD)     # Convection vector previous stage
    
    # Need a container for the computed right hand side vector
    rhs_u = Function(DD).v
    rhs_T = Function(DD).v

In the final solver we will also use bases for dealiasing the nonlinear term,
but we do not add that level of complexity here.

Wall-normal velocity equation
-----------------------------

We implement Eq. :eq:`eq:u2` using the three-stage Runge Kutta equation :eq:`eq:rk3stagesvar`.
To this end we first need to declare some test- and trial functions, as well as
some model constants

.. code-block:: python

    u = TrialFunction(W_BF)
    v = TestFunction(W_BF)
    a = (8./15., 5./12., 3./4.)
    b = (0.0, -17./60., -5./12.)
    c = (0., 8./15., 2./3., 1)
    
    # Specify viscosity and time step size using dimensionless Ra and Pr
    Ra = 10000
    Pr = 0.7
    nu = np.sqrt(Pr/Ra)
    kappa = 1./np.sqrt(Pr*Ra)
    dt = 0.1
    
    # Get one solver for each stage of the RK3
    solver = []
    for rk in range(3):
        mats = inner(div(grad(u)) - ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u)))), v)
        solver.append(chebyshev.la.Biharmonic(*mats))

Notice the one-to-one resemblance with the left hand side of :eq:`eq:rk3stagesvar`, where :math:`\psi^{k+1}`
now has been replaced by :math:`\nabla^2 u` (or ``div(grad(u))``) from Eq. :eq:`eq:u2`.
For each stage we assemble a list of tensor product matrices ``mats``, and in ``chebyshev.la``
there is available a very fast direct solver for exactly this type of (biharmonic)
matrices. The solver is created with ``chebyshev.la.Biharmonic(*mats)``, and here
the necessary LU-decomposition is carried out for later use and reuse on each time step.

The right hand side depends on the solution on the previous stage, and the
convection on two previous stages. The linear part (first term on right hand side of :eq:`eq:rk3stages`)
can be assembled as

.. code-block:: python

    inner(div(grad(u_[0])) + ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u_[0])))), v)

The remaining parts :math:`\frac{\partial^2 H_y}{\partial x \partial y} - \frac{\partial^2 H_x}{\partial y\partial y} + \frac{\partial^2 T}{\partial y^2}`
end up in the nonlinear :math:`\mathcal{N}`. The nonlinear convection term :math:`\boldsymbol{H}` can be computed in many different ways.
Here we will make use of
the identity :math:`(\boldsymbol{u} \cdot \nabla) \boldsymbol{u} = -\boldsymbol{u} \times (\nabla \times \boldsymbol{u}) + 0.5 \nabla\boldsymbol{u} \cdot \boldsymbol{u}`,
where :math:`0.5 \nabla \boldsymbol{u} \cdot \boldsymbol{u}` can be added to the eliminated pressure and as such
be neglected. Compute :math:`\boldsymbol{H} = -\boldsymbol{u} \times (\nabla \times \boldsymbol{u})` by first evaluating
the velocity and the curl in real space. The curl is obtained by projection of :math:`\nabla \times \boldsymbol{u}`
to the no-boundary-condition space ``W_TF``, followed by a backward transform to real space.
The velocity is simply transformed backwards.


.. note::
   If dealiasing is required, it should be used here to create padded backwards transforms of the curl and the velocity,
   before computing the nonlinear term in real space. The nonlinear product should then be forward transformed with
   truncation. To get a space for dealiasing, simply use, e.g., ``W_BF.get_dealiased()``.




.. code-block:: python

    # Get a mask for setting Nyquist frequency to zero
    mask = W_DF.get_mask_nyquist()
    
    def compute_convection(u, H):
        curl = project(Dx(u[1], 0, 1) - Dx(u[0], 1, 1), W_TF).backward()
        ub = u.backward()
        H[0] = W_DF.forward(-curl*ub[1])
        H[1] = W_DF.forward(curl*ub[0])
        H.mask_nyquist(mask)
        return H

Note that the convection has a homogeneous Dirichlet boundary condition in the
non-periodic direction. With convection computed we can assemble :math:`\mathcal{N}`
and all of the right hand side, using the function ``compute_rhs_u``

.. code-block:: python

    def compute_rhs_u(u, T, H, rhs, rk):
        v = TestFunction(W_BF)
        H = compute_convection(u, H)
        rhs[1] = 0
        rhs[1] += inner(v, div(grad(u[0])) + ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u[0])))))
        w0 = inner(v, Dx(Dx(H[1], 0, 1), 1, 1) - Dx(H[0], 1, 2))
        w1 = inner(v, Dx(T, 1, 2))
        rhs[1] += a[rk]*dt*(w0+w1)
        rhs[1] += b[rk]*dt*rhs[0]
        rhs[0] = w0+w1
        rhs.mask_nyquist(mask)
        return rhs
    

Note that we will only use ``rhs`` as a container, so it does not actually matter
which space it has here. We're using ``.v`` to only access the Numpy array view of the Function.
Also note that ``rhs[1]`` contains the right hand side computed at stage ``k``,
whereas ``rhs[0]`` is used to remember the old value of the nonlinear part.

Streamwise velocity
-------------------

The streamwise velocity is computed using Eq. :eq:`eq:div3` and :eq:`eq:vx`. For efficiency we
can here preassemble both matrices seen in :eq:`eq:div3` and reuse them every
time the streamwise velocity is being computed. We will also need the
wavenumber :math:`\boldsymbol{l}`, here retrived using ``W_BF.local_wavenumbers(scaled=True)``.
For :eq:`eq:vx` we preassemble the required Helmholtz solvers, one for
each RK stage.

.. code-block:: python

    # Assemble matrices and solvers for all stages
    B_DD = inner(TestFunction(W_DF), TrialFunction(W_DF))
    C_DB = inner(TestFunction(W_DF), Dx(TrialFunction(W_BF), 0, 1))
    v0 = TestFunction(VD)
    u0 = TrialFunction(VD)
    solver0 = []
    for rk in range(3):
        mats0 = inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*u0 - div(grad(u0)))
        solver0.append(chebyshev.la.Helmholtz(*mats0))
    
    # Allocate work arrays and variables
    u00 = Function(VD)
    b0 = np.zeros((2,)+u00.shape)
    w00 = np.zeros_like(u00)
    dudx_hat = Function(W_DF)
    K = W_BF.local_wavenumbers(scaled=True)[1]
    
    def compute_v(u, rk):
        if comm.Get_rank() == 0:
            u00[:] = u_[1, :, 0].real
        dudx_hat = C_DB.matvec(u[0], dudx_hat)
        with np.errstate(divide='ignore'):
            dudx_hat = 1j * dudx_hat / K
        u[1] = B_DD.solve(dudx_hat, u=u[1])
    
        # Still have to compute for wavenumber = 0
        if comm.Get_rank() == 0:
            b0[1] = inner(v0, 2./(nu*(a[rk]+b[rj])*dt)*Expr(u00) + div(grad(u00)))
            w00 = inner(v0, H_[1, :, 0])
            b0[1] -= (2.*a/nu/(a[rk]+b[rk]))*w00
            b0[1] -= (2.*b/nu/(a[rk]+b[rk]))*b0[0]
            u00 = solver0[rk](u00, b0[1])
            u[1, :, 0] = u00
            b0[0] = w00
        return u

Temperature
-----------

The temperature equation :eq:`eq:T` is implemented using a Helmholtz solver.
The main difficulty with the temperature is the non-homogeneous boundary
condition that requires special attention. A non-zero Dirichlet boundary
condition is implemented by adding two basis functions to the
basis of the function space

.. math::
   :label: _auto12

        
            \phi^D_{N-2} = 0.5(1+x), 
        
        

.. math::
   :label: _auto13

          
            \phi^D_{N-1} = 0.5(1-x),
        
        

with the approximation now becoming

.. math::
   :label: _auto14

        
            T_N(x, y, t) = \sum_{k=0}^{N-1} \sum_{l \in \boldsymbol{l}} \hat{T}_{kl} \phi^D_k(x)\exp(\imath l y), 
        
        

.. math::
   :label: _auto15

          
                         = \sum_{k=0}^{N-3} \sum_{l \in \boldsymbol{l}} \hat{T}_{kl} \phi^D_k(x)\exp(\imath l y) + \sum_{k=N-2}^{N-1} \sum_{l \in \boldsymbol{l}} \hat{T}_{kl} \phi^D_k(x)\exp(\imath l y).
        
        

The boundary condition requires

.. math::
   :label: _auto16

        
        T_N(1, y, t) = \sum_{k=N-2}^{N-1} \sum_{l \in \boldsymbol{l}} \hat{T}_{kl} \phi^D_k(1)\exp(\imath l y), 
        
        

.. math::
   :label: eq:TN0

          
                     = \sum_{l \in \boldsymbol{l}} \hat{T}_{N-2, l} \exp(\imath l y), 
        

and

.. math::
   :label: _auto17

        
        T_N(-1, y, t) = \sum_{k=N-2}^{N-1} \sum_{l \in \boldsymbol{l}} \hat{T}_{kl} \phi^D_k(-1)\exp(\imath l y), 
        
        

.. math::
   :label: eq:TN1

          
                      = \sum_{l \in \boldsymbol{l}} \hat{T}_{N-1, l} \exp(\imath l y). 
        

We find :math:`\hat{T}_{N-2, l}` and :math:`\hat{T}_{N-1, l}` using orthogonality. Multiply :eq:`eq:TN0` and
:eq:`eq:TN1` by :math:`\exp(-\imath m y)` and integrate over the domain :math:`[0, 2\pi]`. We get

.. math::
   :label: _auto18

        
            \hat{T}_{N-2, l} = \int_{0}^{2\pi} T_N(1, y, t) \exp(-\imath l y) dy, 
        
        

.. math::
   :label: _auto19

          
            \hat{T}_{N-1, l} = \int_{0}^{2\pi} T_N(-1, y, t) \exp(-\imath l y) dy.
        
        

Using this approach it is easy to see that any inhomogeneous function :math:`T_N(\pm 1, y, t)`
of :math:`y` and :math:`t` can be used for the boundary condition, and not just a constant.
To implement a non-constant Dirichlet boundary condition, the ``Basis`` function
can take any ``sympy`` function of ``(y, t)``, for exampel by replacing the
creation of ``VT`` by

.. code-block:: python

    import sympy as sp
    y, t = sp.symbols('y,t')
    f = 0.9+0.1*sp.sin(2*(y))*sp.exp(-t)
    VT = Basis(N, family, bc=(0, f))

For merely a constant ``f`` or a ``y``-dependency, no further action is required.
However, a time-dependent approach requires the boundary values to be
updated each time step. To this end there is the function
``BoundaryValues.update_bcs_time``, used to update the boundary values to the new time.
Here we will assume a time-independent boundary condition, but the
final implementation will contain the time-dependent option.

Due to the non-zero boundary conditions there are also a few additional
things to be aware of. Assembling the coefficient matrices will also
assemble the matrices for the two boundary test functions. That is,
for the 1D mass matrix with :math:`u=\sum_{k=0}^{N-1}\hat{T}_k \phi^D_k` and :math:`v=\phi^D_m`,
we will have

.. math::
   :label: _auto20

        
            \left(u, v \right)_w = \left( \sum_{k=0}^{N-1} \hat{T}_k \phi^D_k(x), \phi^D_m \right)_w, 
        
        

.. math::
   :label: _auto21

          
                                 = \sum_{k=0}^{N-3} \left(\phi^D_k(x), \phi^D_m \right)_w \hat{T}_k + \sum_{k=N-2}^{N-1} \left( \phi^D_k(x), \phi^D_m \right)_w \hat{T}_k,
        
        

where the first term on the right hand side is the regular mass matrix for a
homogeneous boundary condition, whereas the second term is due to the non-homogeneous.
Since :math:`\hat{T}_{N-2}` and :math:`\hat{T}_{N-1}` are known, the second term contributes to
the right hand side of a system of equations. All boundary matrices can be extracted
from the lists of tensor product matrices returned by ``inner``. For
the temperature equation these boundary matrices are extracted using
``extract_bc_matrices`` below. The regular solver is placed in the
``solverT`` list, one for each stage of the RK3 solver.

.. code-block:: python

    solverT = []
    lhs_mat = []
    for rk in range(3):
        matsT = inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*p - div(grad(p)))
        lhs_mat.append(extract_bc_matrices([matsT]))
        solverT.append(chebyshev.la.Helmholtz(*matsT))

The boundary contribution to the right hand side is computed for each
stage as

.. code-block:: python

    rhs_T = lhs_mat[rk][0].matvec(T_, rhs_T)

The complete right hand side of the temperature equations can be computed as

.. code-block:: python

    def compute_rhs_T(u, T, rhs, rk):
        q = TestFunction(W_TF)
        rhs[1] = inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*Expr(T)+div(grad(T)))
        rhs[1] -= lhs_mat[rk][0].matvec(T, w0)
        ub = u.backward()
        Tb = T.backward()
        uT_ = BD.forward(ub*Tb)
        w0[:] = 0
        w0 = inner(q, div(uT_), output_array=w0)
        rhs[1] -= (2.*a/kappa/(a[rk]+b[rk]))*w0
        rhs[1] -= (2.*b/kappa/(a[rk]+b[rk]))*rhs[0]
        rhs[0] = w0
        rhs.mask_nyquist(mask)
        return rhs

We now have all the pieces required to solve the Rayleigh Benard problem.
It only remains to perform an initialization and then create a solver
loop that integrates the solution forward in time.

.. code-block:: python

    # initialization
    T_b = Array(W_TF)
    X = W_TF.local_mesh(True)
    T_b[:] = 0.5*(1-X[0]) + 0.001*np.random.randn(*T_b.shape)*(1-X[0])*(1+X[0])
    T_ = T_b.forward(T_)
    T_.mask_nyquist(mask)
    
    def solve(t=0, tstep=0, end_time=1000):
        while t < end_time-1e-8:
            for rk in range(3):
                rhs_u = compute_rhs_u(u_, T_, H_, rhs_u, rk)
                u_[0] = solver[rk](u_[0], rhs_u[1])
                if comm.Get_rank() == 0:
                    u_[0, :, 0] = 0
                u_ = compute_v(u_, rk)
                u_.mask_nyquist(mask)
                rhs_T = compute_rhs_T(u_, T_, rhs_T, rk)
                T_ = solverT[rk](T_, rhs_T[1])
                T_.mask_nyquist(mask)
    
            t += dt
            tstep += 1

A complete solver implemented in a solver class can be found in
`RayleighBenardRk3.py <https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenardRK3.py>`__,
where some of the terms discussed in this demo have been optimized some more for speed.
Note that in the final solver it is also possible to use a :math:`(y, t)`-dependent boundary condition
for the hot wall. And the solver can also be configured to store intermediate results to
an ``HDF5`` format that later can be visualized in, e.g., Paraview. The movie in the
beginning of this demo has been created in Paraview.

.. ======= Bibliography =======

.. bibliography:: papers.bib
   :notcited:
