# gfrp
Generic Fast Randomized Projections


## Third Party
|Dependency | Reference | Comments |
|-|-|-|
|[Blaze](https://bitbucket.org/blaze-lib)|[K. Iglberger, et al.: Expression Templates Revisited: A Performance Analysis of Current Methodologies. SIAM Journal on Scientific Computing, 34(2): C42--C69, 2012](http://epubs.siam.org/sisc/resource/1/sjoce3/v34/i2/pC42_s1)|For optimal performance, this should be linked against BLAS and parallelized, as controlled in blaze/blaze/config/BLAS.h|
|C++17||DenseSVM is currently only tested on gcc under 5.2 and 6.3|
|[Fast Fast-Hadamard Transform](https://github.com/dnbaker/FFHT)|Out-of-place fork of FHT by https://falconn-lib.org/||

