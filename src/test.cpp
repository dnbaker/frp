#include "gfrp/gfrp.h"
#include "gfrp/fsm.h"
#include "FFHT/fht.h"
#include <iostream>
#include "fftw-wrapper/fftw_wrapper.h"
#include <cstdlib>
#include <cstring>
#include <chrono>

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
    CompactRademacher<FLOAT_TYPE> cr(size);
#if 0
    for(size_t i(0); i < cr.size(); ++i)
        std::cerr << "cr at index " << i << " is " << cr[i] << '\n';
#endif
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
    SpinBlockTransformer<CompactRademacher<FLOAT_TYPE>, CompactRademacher<FLOAT_TYPE>, CompactRademacher<FLOAT_TYPE>> spinner(size, size, size, CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size), CompactRademacher<FLOAT_TYPE>(size));
}
