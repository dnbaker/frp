#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>
#include "gfrp/gfrp.h"
#include "FFHT/fht.h"

using namespace std::chrono;

using namespace gfrp;
using namespace gfrp::linalg;
using namespace blaze;

template<typename T>
bool has_neg(const T &mat) {
    for(size_t i(0); i < mat.rows(); ++i)
        for(size_t j(0); j < mat.columns(); ++j)
            if(mat(i, j) < 0) return true;
    return false;
}

int main(int argc, char *argv[]) {
    const std::size_t size(argc <= 1 ? 1 << 16: std::strtoull(argv[1], 0, 10)),
                      niter(argc <= 2 ? 1000: std::strtoull(argv[2], 0, 10));
    CompactRademacher<FLOAT_TYPE> cr(size);
#if 0
    for(size_t i(0); i < cr.size(); ++i)
        std::cerr << "cr at index " << i << " is " << cr[i] << '\n';
    fprintf(stderr, "size: %zu\n", size);
#endif
    std::vector<double> out(size);
    for(size_t j(0); j < out.size(); ++j) out[j] = cr[j];
    const auto ln(log2_64(size));
    {
        Timer time("plain dumb");
        for(size_t i(0); i < niter; ++i)
            fsm::dumb_fht(out.data(), ln);
    }
    std::vector<int8_t> ints(size);
    {
        Timer time("rad");
        for(size_t i(0); i < niter; ++i)
            fsm::rad_fht(cr.template as_type<int8_t>(), &ints[0], ln);
    }
    std::vector<double> out_dumb(size);
    for(size_t i(0); i < size; ++i) out_dumb[i] = cr[i];
    {
        Timer time("FFHT::fht");
        for(size_t i(0); i < niter; ++i) {
            //for(size_t j(0); j < out_dumb.size(); ++j) out[j] = cr[j];
            fht(out_dumb.data(), ln);
        }
    }
    std::mt19937_64 gen(0);
    std::shuffle(out.begin(), out.end(), gen);
    aes::AesCtr aesgen(0);
    std::shuffle(out.begin(), out.end(), aesgen);
    ScalingBlock<FLOAT_TYPE> sb(size);
    blaze::DynamicVector<uint64_t> tvec(size); // Full vector
    blaze::DynamicVector<uint64_t> tmul(size); // Full vector
    blaze::DynamicVector<uint64_t> tvout(size);
#if 0
    PRNVector<std::mt19937_64, std::normal_distribution<float>> prn_vec(size, 0);
#else
    PRNVector<aes::AesCtr<uint64_t>> prn_vec(size, 0);
    auto t = prn_vec[14];
    std::cerr << t << '\n';
#endif
    for(auto &el: tvec) {
#if 0
        el = (float)std::rand() / RAND_MAX;
#else
        el = (uint64_t(std::rand()) << 32) | std::rand();
#endif
    }
    size_t i(0);
    for(auto el: prn_vec) {
        tmul[i++] = el;
    }
    {
        Timer time("Pre-computed.");
        for(size_t j(0); j < niter; ++j) {
            for(size_t i(0), e(tvec.size()); i < e; ++i) {
                tvout[i] = tvec[i] * tmul[i];
            }
        }
    }
    {
        Timer time("On-the-fly.");
        for(size_t j(0); j < niter; ++j) {
            auto prn_it(prn_vec.begin());
            for(size_t i(0), e(tvec.size()); i < e; ++i, ++prn_it) {
                tvout[i] = tvec[i] * (*prn_it);
            }
        }
    }
    {
        Timer time("On-the-fly, all iterator.");
        for(size_t j(0); j < niter; ++j) {
            auto prn_it(prn_vec.begin());
            for(auto tvo(tvout.begin()), tve(tvout.end()), tvc(tvec.begin());
                tvo != tve; ++tvo, ++tvc, ++prn_it) {
                *tvo = *tvc * (*prn_it);
            }
        }
    }
    i = 0;
    for(auto el: prn_vec) {
        tmul[i++] = el;
    }
    i = 0;
    for(auto el: prn_vec) {
        tvout[i++] = el;
    }
    i = 0;
    for(auto el: prn_vec) {
        tvec[i++] = el;
    }
    //fprintf(stderr, "First entries: %zu, %zu, %zu\n", tvout[0], tvec[0], *prn_vec.begin());
    for(auto ie(tvec.begin()), io(tvout.begin()); ie != tvec.end(); ++ie, ++io) {
        if(*ie != *io) {
            std::cerr << "Invec: \n\n\n" << tvec;
            std::cerr << "Outvec: \n\n\n" << tvout;
            assert(false);
        }
    }
    using SpinBlockType = SpinBlockTransformer<CompactRademacher<FLOAT_TYPE>, CompactRademacher<FLOAT_TYPE>, CompactRademacher<FLOAT_TYPE>, HadamardBlock<FLOAT_TYPE>, GaussianScalingBlock<FLOAT_TYPE>>;
    auto tuple = std::make_tuple(CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), HadamardBlock<FLOAT_TYPE>(size), GaussianScalingBlock<FLOAT_TYPE>(1337));
    SpinBlockType spinner(size, size, size, std::move(tuple));
    JLTransform<blaze::DynamicMatrix<float>> jlt(24, 1000);
    jlt.fill(1337);
    auto sizes(mach::get_cache_sizes());
    std::cerr << sizes.str() << '\n';
    blaze::DynamicVector<float> v1(1000);
    v1 += float(4.);
    for(size_t i(0); i < niter; ++i)
    {
        // Test random-access is working.
        aes::AesCtr<uint64_t> rng;
        std::vector<uint64_t> seq;
        seq.reserve(size);
        for(size_t i(0); i < size; ++i) seq.push_back(rng());
        std::vector<uint64_t> ram;
        ram.reserve(size);
        for(size_t i(0); i < size; ram.push_back(rng[i++]));
        for(size_t i(0); i < size; ++i) {
            assert(seq[i] == ram[i]);
        }
    }
#if 0
    gaussian_fill(tvec);
    SpinBlockType spinner(size, size, size, std::tuple{CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size)});
    {
        Timer time("FFHT::fht");
        for(size_t i(0); i < niter; ++i) {
            //for(size_t j(0); j < out_dumb.size(); ++j) out[j] = cr[j];
            fht(out_dumb.data(), ln);
        }
    }
#endif
}
