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

using KernelBase = kernel::rf::KernelBlock<FLOAT_TYPE>;
using KernelType = kernel::Kernel<KernelBase, kernel::GaussianFinalizer>;
using ORFKernelBase = kernel::orf::KernelBlock<FLOAT_TYPE>;
using ORFKernelType = kernel::Kernel<ORFKernelBase, kernel::GaussianFinalizer>;
using SORFKernelBase = kernel::sorf::KernelBlock<FLOAT_TYPE>;
using SORFKernelType = kernel::Kernel<SORFKernelBase, kernel::GaussianFinalizer>;

struct GaussianKernel {
    template<typename V1, typename V2>
    double operator()(const V1 &v1, const V2 &v2, double sigma) {
        double dist(dot(v1 - v2, v1 - v2));
        assert(dist >= 0.);
        auto xp = -dist / (2. * sigma * sigma);
        return std::exp(xp);
    }
};

int usage(char *arg) {
    std::fprintf(stderr, "Usage: %s <opts>\n"
                         "-i\tInput size [128]\n-s:sigma [1.0]\n-SOutput size [4096]\n-n: nsample points\n", arg);
    return EXIT_FAILURE;
}

template<typename Mat1, typename Mat2, typename KernelType>
void time_stuff(Mat1 &outm, const Mat2 &in, const char *taskname, const KernelType &kernel) {
    const size_t nrows(in.rows()), insize(in.columns()), outsize(outm.columns());
    Timer time(std::string("How long to apply kernel ") + taskname + " times on dimensions " + std::to_string(insize) + ", " + std::to_string(outsize) + ".");
    for(size_t i(0); i < nrows; ++i) {
        auto orow(row(outm, i));
        kernel.apply(orow, row(in, i));
    }
}

int main(int argc, char *argv[]) {
    int c;
    size_t insize(1 << 6), outsize(1 << 14), nrows(100);
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
    KernelType kernel(outsize, insize, 1337, sigma);
    SORFKernelType sorfkernel(outsize, insize, 1337 * 3, sigma);
    ORFKernelType orfkernel(outsize, insize, 1337 * 2, sigma);
    blaze::DynamicMatrix<FLOAT_TYPE> outm(nrows, outsize << 1), outmorf(nrows, outsize << 1), outmsorf(nrows, outsize << 1);
    blaze::DynamicMatrix<FLOAT_TYPE> in(nrows, insize);
    size_t seed(0);
    //omp_set_num_threads(6);
    //#pragma omp parallel for
    for(size_t i(0); i < nrows; ++i) {
        auto inrow(row(in, i));
        unit_gaussian_fill(inrow, seed + i);
        inrow *= 1./norm(inrow);
        //const FLOAT_TYPE val(std::sqrt(meanvar(row(in, i)).second) * i);
        //auto r(row(in, i));
        //vec::blockadd(r, val);
    }
    blaze::DynamicMatrix<FLOAT_TYPE> indists(nrows, nrows);
    blaze::DynamicMatrix<FLOAT_TYPE> outdists(nrows, nrows);
    blaze::DynamicMatrix<FLOAT_TYPE> outdistsorf(nrows, nrows);
    blaze::DynamicMatrix<FLOAT_TYPE> outdistssorf(nrows, nrows);
    GaussianKernel gk;
    for(size_t i(0), j; i < nrows; ++i)
        for(indists(i, i) = 1e-300, j = i + 1; j < nrows; ++j)
             indists(i, j) = indists(j, i) = gk(row(in, i), row(in, j), sigma);
    {
        time_stuff(outm, in, "rf", kernel);
        time_stuff(outmorf, in, "orf", orfkernel);
        time_stuff(outmsorf, in, "sorf", sorfkernel);
    }
    for(size_t i(0), j; i < nrows; ++i)
        for(outdists(i, i) = 1e-300, j = i + 1; j < nrows; ++j)
            outdists(i, j) = outdists(j, i) = dot(row(outm, i), row(outm, j)),
            outdistsorf(i, j) = outdistsorf(j, i) = dot(row(outmorf, i),  row(outmorf, j)),
            outdistssorf(i, j) = outdistssorf(j, i) = dot(row(outmsorf, i),  row(outmsorf, j));
    //std::cerr << "Input full kernel distances: " << indists << '\n';
    //std::cerr << "Input approx kernel distances: " << outdists << '\n';
    blaze::DynamicMatrix<FLOAT_TYPE> ratios(nrows, nrows);
    for(size_t i(0); i < nrows; ++i)
            for(size_t j(0); j < nrows; ++j) ratios(i, j) = outdists(i, j) / indists(i, j);
    //std::cerr << "Output ratios: " << ratios << '\n';
    std::cout << "#Ratio, Gaussian Distances, RF Approx, ORF Approx, SORF Approx\n";
    for(size_t i(0); i < nrows; ++i)
            for(size_t j(0); j < nrows; ++j)
                std::cout << ratios(i, j) << ", " << indists(i, j) << ", " << outdists(i, j) << ", "
                          << outdistsorf(i, j) << ", " << outdistssorf(i, j) << '\n';
    std::cerr << "Successfully completed " << *argv << '\n';
#if 0
    std::cerr << in << '\n';
    std::cerr << outm << '\n';
#endif
}
