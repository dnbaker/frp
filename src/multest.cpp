#include "gfrp/gfrp.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std::chrono;
using namespace gfrp;
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
    const unsigned len(argc == 1 ? 1 << 16 : std::atoi(argv[1]));
    DynamicVector<FLOAT_TYPE> vec(len);
    DynamicVector<FLOAT_TYPE> ret(len);
    for(auto &el: vec) el = FLOAT_TYPE(std::rand()) / RAND_MAX;
}
