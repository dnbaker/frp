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
using FFKernelBase = kernel::ff::KernelBlock<FLOAT_TYPE>;
using FFKernelType = kernel::Kernel<FFKernelBase, kernel::GaussianFinalizer>;

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
double time_stuff(Mat1 &outm, const Mat2 &in, const char *taskname, const KernelType &kernel, double sigma) {
    const size_t nrows(in.rows()), insize(in.columns()), outsize(outm.columns());
    Timer time(std::string(taskname) + " " + std::to_string(nrows) + " times on dimensions " + std::to_string(insize) + ", " + std::to_string(outsize) + " and sigma = " + std::to_string(sigma) + ".");
    for(size_t i(0); i < nrows; ++i) {
        auto orow(row(outm, i));
        kernel.apply(orow, row(in, i));
    }
    return time.time();
}

int main(int argc, char *argv[]) {
    int c;
    size_t insize(1 << 6), outsize(1 << 14), nrows(250);
    double sigma(1.);
    bool override(false);
    while((c = getopt(argc, argv, "n:i:S:e:M:s:p:b:l:o:5OBrh?")) >= 0) {
        switch(c) {
            case 'i': insize = std::strtoull(optarg, 0, 10); break;
            case 's': sigma = std::atof(optarg); break;
            case 'S': outsize = std::strtoull(optarg, 0, 10); break;
            case 'n': nrows = std::strtoull(optarg, 0, 10); break;
            case 'O': override = true; break;
            case 'h': case '?': usage: return usage(*argv);
        }
    }
    if(argc > optind) goto usage;
    outsize = roundup(outsize);
    insize = roundup(insize);
    SORFKernelType sorfkernel(outsize, insize, 1337 * 3, sigma);
    FFKernelType ffkernel(outsize, insize, 1337 * 4, sigma);
    blaze::DynamicMatrix<FLOAT_TYPE> outm(nrows, outsize << 1);
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
#if 0
    GaussianKernel gk;
    for(size_t i(0), j; i < nrows; ++i)
        for(indists(i, i) = 1e-300, j = i + 1; j < nrows; ++j)
             indists(i, j) = indists(j, i) = gk(row(in, i), row(in, j), sigma);
#endif
    double times[4]{0};
    {
        if((insize * outsize) < (5000 * 32000) || override) {
            {
            KernelType kernel(outsize, insize, 1337, sigma);
            times[0] = time_stuff(outm, in, "rf", kernel, sigma);
            }
            ORFKernelType orfkernel(outsize, insize, 1337 * 2, sigma);
            times[1] = time_stuff(outm, in, "orf", orfkernel, sigma);
        }
        times[2] = time_stuff(outm, in, "sorf", sorfkernel, sigma);
        times[3] = time_stuff(outm, in, "ff", ffkernel, sigma);
    }
    std::vector<std::string> names {"rf", "orf", "sorf", "ff"};
    for(size_t i(0); i < 4; ++i) std::fprintf(stdout, "%s\t", names[i].data());
    std::fputc('\n', stdout);
    for(size_t i(0); i < 4; ++i) std::fprintf(stdout, "%lf\t", times[i]);
    std::fputc('\n', stdout);
}
