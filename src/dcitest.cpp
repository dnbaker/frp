#include "include/frp/dci.h"
#include <iostream>

using namespace frp;
using namespace dci;
int main() {
    int nd = 4;
    DCI<blaze::DynamicVector<float>> dci(10, 4, nd);
    std::cerr << "made dci\n";
    std::vector<blaze::DynamicVector<float>> ls;
    std::mt19937_64 mt;
    for(size_t i = 0; i < 100; ++i) {
        ls.emplace_back(nd);
        for(auto &x: ls.back())
            x = std::ldexp(double(mt()), 64);
    }
    for(const auto &v: ls)
        dci.add_item(v);
    dci.query(ls[0], 3);
    std::cerr << "added item to dci\n";
}
