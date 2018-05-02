"""Module for optimized Cython functions

Some methods performed in Python may be slowing down solvers. In this optimization
module we place optimized Cython functions that are to be used instead of default
Python methods. Some methods are implemented solely in Cython and only called
from withing the regular Python modules.

"""
