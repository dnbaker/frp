#include "gfrp/gfrp.h"
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
template<typename T>
bool has_vneg(const T& vec) {for(const auto &el: vec){ if(el < 0) return true;} return false;}

int main(int argc, char *argv[]) {
    const std::size_t size(argc <= 1 ? 1 << 20: std::strtoull(argv[1], 0, 10));
    CompactRademacher cr(1 << 16);
    for(size_t i(0); i < cr.size(); ++i) std::cerr << "cr at index " << i << " is " << cr[i] << '\n';
}
