# frp: Fast Randomized Projections
We use [Blaze](https://bitbucket.org/blaze-lib) for fast linear algebra, [Sleef](https://github.com/shibatch/sleef) for fast trigonometric operations,
[Fast Fast-Hadamard Transform](https://github.com/dnbaker/FFHT) from FALCONN-LIB for the Fast Hadamard Transform, [FFTW3](http://fftw.org/) for the FFT, and [boost](https://github.com/boostorg) for
special functions and random number generators. Only required boost headers are provided as submodules and no installation of boost is required.

## Contents
0. Orthogonal JL transform with linear space and linearithmic runtime
    1. This is available through the `ojlt` executable, in C++ programs accessing include/frp/jl.h, and using python bindings by `cd py && make`.
1. Kernel projections
    1. We support kernel approximation for the Gaussian kernel using Random Fourier Features, Orthogonal Random Features, Structured Orthogonal Random Features, and FastFood.
    2. We recommend Structured Orthogonal Random Features, as it has the highest accuracy in our experiments and can also be hundreds of times faster while still having a small memory footprint.
2. A type-generic SIMD interface (vec/vec.h), which abstracts operations to allow the compiler to use the widest vectors possible as needed, facilitating generically dispatching the fastest implementation possible on a machine.
3. Utilities
    2. PRNVector (PseudoRandom Number Vector) to provide access to random vectors using only constant memory requires instead of explicitly storing them by generating values as needed.
    3. Utilities for sampling and filling containers from distributions.
    4. Acquiring cache sizes from the OS.
4. Linear algebra methods
    1. Implementation of the Gram-Schmidt algorithm for orthogonalizing matrices.
    2. PCA using full eigendecomposition for symmetric matrices.
    3. Covariance Matrix calculation
5. Miscellaneous, related work
    1. Dynamic Continuous Indexing for real-valued data
        1. [Dynamic Continuous Indexing](https://arxiv.org/abs/1512.00442)
          1. Tested
        2. [Prioritized DCI](https://arxiv.org/abs/1703.00440)
          2. Draft form.

### Build instructions

`make` should compile a variety of tests.
We assume you're using BLAS for your linear algebra; to avoid doing that, modify the Makefile and remove the `-DBLAZE*` flags.

To specify a different blas header file, use the CBLASFILE variable when compiling:
```bash
make ojlt CBLASFILE=mkl_cblas.h
# Or, use an environmental variable
export CBLASFILE=mkl_cblas.h && \
make ojlt
```

        

## Commentary

The initial design of this library was to implement methods from [https://arxiv.org/abs/1703.00864](https://arxiv.org/abs/1703.00864). The core transformss on which it is built are fast fast-hadamard transform accelerated structured matrix vector products. This has applications in memory-efficient, accelerated Johnson-Lindenstrauss Transforms, gaussian kernel approximation for linearizing datasets and FastFood/Adaptive Random Spinners.

## DCI/Prioritized DCI usage

Notes:

During construction, it may be advantageous to use a std::set to maintain sorted indexes (logarithmic update time), whereas at query time, it's faster to use a contiguous array.
We provide the cvt function, which copies the index, but converts the sorted index type from what it used to be (usually a red-black tree) into the destination type,
by default an always-sorted array.

We suggest doing this for the purposes of faster construction and faster queries.

Additionally, we do not store any points, just references to them.
