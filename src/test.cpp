#include "gfrp/gfrp.h"
#include "FFHT/fht.h"
#include <iostream>
#include "fftw-wrapper/fftw_wrapper.h"
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>

using namespace std::chrono;

using namespace gfrp;
using namespace blaze;

template<typename T>
bool has_neg(const T &mat) {
    for(size_t i(0); i < mat.rows(); ++i)
        for(size_t j(0); j < mat.columns(); ++j)
            if(mat(i, j) < 0) return true;
    return false;
}

class Timer {
    using TpType = std::chrono::system_clock::time_point;
    std::string name_;
    TpType start_, stop_;
public:
    Timer(std::string &&name=""): name_{name}, start_(std::chrono::system_clock::now()) {}
    void stop() {stop_ = std::chrono::system_clock::now();}
    void restart() {start_ = std::chrono::system_clock::now();}
    void report() {std::fprintf(stderr, "About to report\n"); std::cerr << "Took " << std::chrono::duration<double>(stop_ - start_).count() << "s for task '" << name_ << "'\n";}
    ~Timer() {stop(); /* hammertime */ report();}
};

template<typename T>
bool has_vneg(const T& vec) {for(const auto &el: vec){ if(el < 0) return true;} return false;}

int main(int argc, char *argv[]) {
    const std::size_t size(argc <= 1 ? 1 << 16: std::strtoull(argv[1], 0, 10)),
                      niter(argc <= 2 ? 1000: std::strtoull(argv[2], 0, 10));
#if 0
    CompactRademacher<FLOAT_TYPE> cr(size);
    for(size_t i(0); i < cr.size(); ++i)
        std::cerr << "cr at index " << i << " is " << cr[i] << '\n';
    fprintf(stderr, "size: %zu\n", size);
    std::vector<double> out(size);
    fft::FFTWDispatcher<double> disp(out.size(), false, false, fft::tx::R2HC);
    disp.make_plan(out.data(), out.data());
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
            //fsm::rad_fht(cr, &ints[0], ln);
            disp.run(out.data(), out.data());
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
    using SpinBlockType = SpinBlockTransformer<CompactRademacher<FLOAT_TYPE>, CompactRademacher<FLOAT_TYPE>, CompactRademacher<FLOAT_TYPE>>;
    SpinBlockType spinner(size, size, size, CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size));
    std::mt19937_64 gen(0);
    std::shuffle(out.begin(), out.end(), gen);
    aes::AesCtr aesgen(0);
    std::shuffle(out.begin(), out.end(), aesgen);
    ScalingBlock<FLOAT_TYPE> sb(size);
#endif
#if 0
    blaze::DynamicVector<float> tvec(size); // Full vector
    blaze::DynamicVector<float> tmul(size); // Full vector
    blaze::DynamicVector<float> tvout(size);
#else
    blaze::DynamicVector<uint64_t> tvec(size); // Full vector
    blaze::DynamicVector<uint64_t> tmul(size); // Full vector
    blaze::DynamicVector<uint64_t> tvout(size);
#endif
#if 0
    PRNVector<std::mt19937_64, std::normal_distribution<float>> prn_vec(size, 0);
#else
    PRNVector<aes::AesCtr, UnchangedRNGDistribution<aes::AesCtr>> prn_vec(size, 0);
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
    //SpinBlockType spinner(size, size, size, std::tuple{CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), HadamardBlock<FLOAT_TYPE>(size), GaussianScalingBlock<FLOAT_TYPE>(1337)});
    JLTransform<blaze::DynamicMatrix<float>> jlt(24, 1000);
    jlt.fill(1337);
    auto sizes(mach::get_cache_sizes());
    std::cerr << sizes.str() << '\n';
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
