#include "gfrp/gfrp.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std::chrono;

using namespace gfrp::tx;
using namespace blaze;

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

int main(int argc, char *argv[]) {
    const size_t len(argc == 1 ? 1 << 16 : std::atoi(argv[1]));
    const size_t niter(argc <= 2 ? 1 << 13: std::atoi(argv[2]));
    DynamicVector<FLOAT_TYPE> vec(len);
    DynamicVector<FLOAT_TYPE> ret(len);
    for(auto &el: vec) el = FLOAT_TYPE(std::rand()) / RAND_MAX;
    std::cerr << "Int mults:";
    auto radint(make_rademacher<DynamicVector<int>>(len));
    {
        Timer timer;
        for(size_t i(0); i < niter; ++i) {
            ret = vec * radint;
        }
    }
    std::cerr << "Float mults:";
    auto radf(make_rademacher<DynamicVector<FLOAT_TYPE>>(len));
    {
        Timer timer;
        for(size_t i(0); i < niter; ++i) {
            ret = vec * radf;
        }
    }
}
