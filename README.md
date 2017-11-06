# gfrp
Generic Fast Randomized Projections

## Third Party
|Dependency | Reference | Comments |
|-|-|-|
|[Blaze](https://bitbucket.org/blaze-lib)|[K. Iglberger, et al.: Expression Templates Revisited: A Performance Analysis of Current Methodologies. SIAM Journal on Scientific Computing, 34(2): C42--C69, 2012](http://epubs.siam.org/sisc/resource/1/sjoce3/v34/i2/pC42_s1)|For optimal performance, this should be linked against BLAS and parallelized, as controlled in blaze/blaze/config/BLAS.h|
|C++17||DenseSVM is currently only tested on gcc under 5.2 and 6.3|
|[Fast Fast-Hadamard Transform](https://github.com/dnbaker/FFHT)|Out-of-place fork of FHT by https://falconn-lib.org/||

## TODO


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

## TODONE
1. Add Hadamard Fourier Features
2. Add DCT Fourier Features
3. Random Rademacher and other probability distribution arrays
  1. Both precomputed and PRN-generated.


