#include "gfrp/gfrp.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std::chrono;
using namespace gfrp;
using namespace blaze;

template<typename T>
void print_vec(T &vec) {
    std::cerr << std::scientific;
    std::cerr << "[";
    for(auto el: vec) std::cerr << el << ",";
    std::cerr << "]\n";
}


int main(int argc, char *argv[]) {
    const unsigned len(argc == 1 ? 1 << 16 : std::atoi(argv[1]));
    DynamicVector<FLOAT_TYPE> vec(len);
    DynamicVector<FLOAT_TYPE> ret(len);
    for(auto &el: vec) el = FLOAT_TYPE(std::rand()) / RAND_MAX;
    std::cerr << "Making vec\n";
    PRNVector<aes::AesCtr<uint64_t>,
              unit_normal<FLOAT_TYPE>> pv(len);
    std::cerr << "Made \n";
    auto it(vec.begin());
    unsigned i(0);
    for(auto el: pv) {
        std::cerr << "Accessing index " << i + 1;
         *it = el;
        std::cerr << "pv[" << i << "] is " << pv[i] << '\n';
        ++it;
        ++i;
    }
}
