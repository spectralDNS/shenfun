cdef int get_number_of_blocks(int level, long* D, int N)
cdef int get_h(int level, long* D, int N)
cdef int get_number_of_submatrices(long* D, int L)
cdef void get_ij(int* ij, int level, int block, int diags, int h, int s, long* D, int L)
