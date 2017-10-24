#include "gfrp/gfrp.h"
#include "gfrp/fsm.h"
#include "FFHT/fht.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std::chrono;

using namespace gfrp::dist;
using namespace gfrp::tx;
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
    TpType start_, stop_;
public:
    Timer(): start_(std::chrono::system_clock::now()) {}
    void stop() {stop_ = std::chrono::system_clock::now();}
    void restart() {start_ = std::chrono::system_clock::now();}
    void report() {std::cerr << "Took " << std::chrono::duration<double>(stop_ - start_).count() << "s\n";}
    ~Timer() {stop(); /* hammertime */ report();}
};

template<typename T>
bool has_vneg(const T& vec) {for(const auto &el: vec){ if(el < 0) return true;} return false;}

int main(int argc, char *argv[]) {
    const std::size_t size(argc <= 1 ? 1 << 16: std::strtoull(argv[1], 0, 10)),
                      niter(argc <= 2 ? 1000: std::strtoull(argv[2], 0, 10));
    CompactRademacher<uint64_t, double> cr(size);
    random_fill(cr.data(), cr.nwords());
    for(size_t i(0); i < cr.size(); ++i)
        std::cerr << "cr at index " << i << " is " << cr[i] << '\n';
    std::vector<double> out(size);
    for(size_t j(0); j < out.size(); ++j) out[j] = cr[j];
    auto ln([](size_t n){auto ret(0); while(n>>=1) ++ret; return ret;}(size));
    {
        Timer time;
        for(size_t i(0); i < niter; ++i)
            fsm::dumb_fht(out.data(), ln);
    }
    std::vector<double> out_dumb(size);
    for(size_t i(0); i < 1 << 16; ++i) out_dumb[i] = cr[i];
    {
        Timer time;
        for(size_t i(0); i < niter; ++i) {
            //for(size_t j(0); j < out_dumb.size(); ++j) out[j] = cr[j];
            fht(out_dumb.data(), ln);
        }
    }
}
