# cython: language_level = 3str

cdef extern from "fastgl.cpp":
    pass

cdef extern from "fastgl.h" namespace "fastgl":

    ctypedef struct QuadPair:
        double theta
        double weight
        double x()

    cdef QuadPair GLPair(int, int)
