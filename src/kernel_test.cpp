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
    const std::size_t size(argc <= 1 ? 1 << 10: std::strtoull(argv[1], 0, 10)),
                   outsize(argc <= 2 ? 1 << 13: std::strtoull(argv[1], 0, 10)),
                     niter(argc <= 3 ? 1000: std::strtoull(argv[2], 0, 10));
    KernelType kernel(outsize, size, 1., 1337);
    blaze::DynamicVector<FLOAT_TYPE> out(outsize << 1);
    blaze::DynamicVector<FLOAT_TYPE> in(size);
    ff::GaussianFinalizer gf;
    blaze::DynamicVector<FLOAT_TYPE> dv(1 << 5);
    for(size_t i(0); i < 1 << 4; ++i) {
        dv[i] = std::acos(2. * M_PI / (1 << 4));
    }
    std::vector<int> iota(16);
    std::iota(iota.begin(), iota.end(), 0);
    ks::string str(ks::sprintf("Before shuffle: "));
    ksprint(iota, str);
    OnlineShuffler tmp(133);
    tmp.apply(iota);
    str.puts("\nAfter shuffle:\n");
    ksprint(iota, str);
    str.write(stderr);
    str.clear(); str.puts("Before: ");
    ksprint(dv, str);
    str.sprintf("\nAfter: ");
    gf.apply(dv);
    gf.apply(dv);
    ksprint(dv, str);
    str.write(stderr); str.clear();
    unit_gaussian_fill(in, 1338);
    ksprint(in, str);
    std::fprintf(stderr, "in: '%s'\n", str.data());
    str.clear();
    kernel.apply(out, in);
    ksprint(out, str);
    std::fprintf(stderr, "out: '%s'\n", str.data());
}
