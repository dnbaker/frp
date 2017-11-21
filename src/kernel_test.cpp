#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>
#include "gfrp/gfrp.h"

using namespace std::chrono;

using namespace gfrp;
using namespace gfrp::linalg;
using namespace blaze;

using KernelBase = ff::FastFoodKernelBlock<FLOAT_TYPE>;
using KernelType = ff::Kernel<KernelBase, ff::GaussianFinalizer>;

int main(int argc, char *argv[]) {
    const std::size_t size(argc <= 1 ? 1 << 8: std::strtoull(argv[1], 0, 10)),
                   outsize(argc <= 2 ? 1 << 10: std::strtoull(argv[1], 0, 10)),
                     niter(argc <= 3 ? 1000: std::strtoull(argv[2], 0, 10));
    KernelType kernel(outsize, size, 1., 1337);
    blaze::DynamicVector<FLOAT_TYPE> out(outsize << 1);
    blaze::DynamicVector<FLOAT_TYPE> in(size);
    unit_gaussian_fill(in, 1338);
    ks::string str;
    ksprint(in, str);
    std::fprintf(stderr, "in: '%s'\n", str.data());
    str.clear();
    kernel.apply(out, in);
    ksprint(out, str);
    std::fprintf(stderr, "out: '%s'\n", str.data());
}
