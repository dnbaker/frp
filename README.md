# gfrp
Generic Fast Randomized Projections
We use [Blaze](https://bitbucket.org/blaze-lib) for fast linear algebra, [Sleef](https://github.com/shibatch/sleef) for fast trigonometric operations,
[Fast Fast-Hadamard Transform](https://github.com/dnbaker/FFHT) from FALCONN-LIB for the Fast Hadamard Transform, [FFTW3](http://fftw.org/) for the FFT, and [boost](https://github.com/boostorg) for
special functions and random number generators. Only required boost headers are provided as submodules and no installation of boost is required.

## Contents
1. Orthogonal JL transform with linear space and linearithmic runtime
    1. This is available through the `ojlt` executable, in C++ programs accessing include/frp/jl.h, and using python bindings by `cd py && make`.
2. Kernel projections
    1. These are in development and largely do not work. The src/*cpp files are currently all testing grounds.
3. A type-generic SIMD interface (include/frp/vec.h), which abstracts operations to allow the compiler to use the widest vectors possible as needed, facilitating generically dispatching the fastest implementation possible on a machine.
4. Utilities
    1. Templated SIMD-based and unroll AES-CTR, based on the implementation used in Lemire's testingRNG repository.
    2. PRNVector to provide access to random vectors using only constant memory requires instead of explicitly storing them.
    3. Utilities for sampling and filling containers from distributions.
    4. Acquiting cache sizes from the OS.
    5. Implementation of the Gram-Schmidt algorithm for orthogonalizing matrices.


1. Kernels:
    1. Gaussian [Default, see Recht and Rahimi]
    2. Laplacian [ibid.]
    3. Cauchy [ibid.]
    4. Angular [See arxiv 1703.00864]
    5. Dot Product [See arxiv 1407.5599, table 1 for the rest]
    6. Polynomial
    7. Hellinger
    8. Chi Squared
    9. Intersection 
    10. Jenson-Shannon
    11. Skewed-Chi Squared
    12. Skewed-Intersection
    13. Exponential Semigroup
    14. Reciprocal Semingroup
    15. Arc-Cosine
