#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>
#include <omp.h>
#include "frp/frp.h"

using namespace std::chrono;

using namespace frp;
using namespace frp::linalg;
using namespace blaze;

using KernelBase = ff::FastFoodKernelBlock<FLOAT_TYPE>;
using KernelType = ff::Kernel<KernelBase, ff::GaussianFinalizer>;

struct GaussianKernel {
    template<typename V1, typename V2>
    double operator()(const V1 &v1, const V2 &v2, double sigma) {
        auto xp = -dot(v1 - v2, v1 - v2) / (2. * sigma);
        return std::exp(xp);
    }
};

int usage(char *arg) {
    std::fprintf(stderr, "Usage: %s <opts>\n"
                         "-i\tInput size [128]\n-s:sigma [1.0]\n-SOutput size [4096]\n-n: nsample points\n", arg);
    return EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
    int c;
    size_t insize(1 << 7), outsize(1 << 12), nrows(100);
    double sigma(1.);
    while((c = getopt(argc, argv, "n:i:S:e:M:s:p:b:l:o:5Brh?")) >= 0) {
        switch(c) {
            case 'i': insize = std::strtoull(optarg, 0, 10); break;
            case 's': sigma = std::atof(optarg); std::fprintf(stderr, "sigma is %lf\n", sigma); break;
            case 'S': outsize = std::strtoull(optarg, 0, 10); break;
            case 'n': nrows = std::strtoull(optarg, 0, 10); break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }
    if(argc > optind) goto usage;
    outsize = roundup(outsize);
    std::fprintf(stderr, "nrows: %zu. insize: %zu. outsize: %zu. sigma: %le\n", nrows, insize, outsize, sigma);
    KernelType kernel(outsize, insize, sigma, 1337);
    blaze::DynamicMatrix<FLOAT_TYPE> outm(nrows, outsize << 1);
    blaze::DynamicMatrix<FLOAT_TYPE> in(nrows, insize);
    size_t seed(0);
    //omp_set_num_threads(6);
    //#pragma omp parallel for
    for(size_t i(0); i < nrows; ++i) {
        auto inrow(row(in, i));
        unit_gaussian_fill(inrow, seed + i);
        //const FLOAT_TYPE val(std::sqrt(meanvar(row(in, i)).second) * i);
        //auto r(row(in, i));
        //vec::blockadd(r, val);
    }
    blaze::DynamicMatrix<FLOAT_TYPE> indists(nrows, nrows);
    blaze::DynamicMatrix<FLOAT_TYPE> outdists(nrows, nrows);
    GaussianKernel gk;
    for(size_t i(0), j; i < nrows; ++i)
        for(indists(i, i) = 1e-300, j = i + 1; j < nrows; ++j)
             indists(i, j) = indists(j, i) = gk(row(in, i), row(in, j), sigma);
    {
        Timer time(std::string("How long to apply kernel ") + std::to_string(nrows) + " times on dimensions " + std::to_string(insize) + ", " + std::to_string(outsize) + ".");
        for(size_t i(0); i < nrows; ++i) {
            auto orow(row(outm, i));
            kernel.apply(orow, row(in, i));
        }
    }
    for(size_t i(0), j; i < nrows; ++i)
        for(outdists(i, i) = 1e-300, j = i + 1; j < nrows; ++j)
            outdists(i, j) = outdists(j, i) = dot(row(outm, i) - row(outm, j), row(outm, i) - row(outm, j));
    //std::cerr << "Input full kernel distances: " << indists << '\n';
    //std::cerr << "Input approx kernel distances: " << outdists << '\n';
    blaze::DynamicMatrix<FLOAT_TYPE> ratios(nrows, nrows);
    for(size_t i(0); i < nrows; ++i)
            for(size_t j(0); j < nrows; ++j) ratios(i, j) = outdists(i, j) / indists(i, j);
    //std::cerr << "Output ratios: " << ratios << '\n';
    std::cout << "#Ratio, Gaussian Distances, Approx Distances\n";
    for(size_t i(0); i < nrows; ++i)
            for(size_t j(0); j < nrows; ++j)
                std::cout << ratios(i, j) << ", " << indists(i, j) << ", " << outdists(i, j) << '\n';
#if 0
    std::cerr << in << '\n';
    std::cerr << outm << '\n';
#endif
}
