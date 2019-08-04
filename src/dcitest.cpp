#include "include/frp/dci.h"
#include <iostream>

using namespace frp;
using namespace dci;
int main() {
    int nd = 40, npoints = 1000;
    DCI<blaze::DynamicVector<float>> dci(20, 120, nd);
    //DCI<blaze::DynamicVector<float>> dci2(10, 4, nd, 1e-5, true);
    std::cerr << "made dci\n";
    std::vector<blaze::DynamicVector<float>> ls;
    std::mt19937_64 mt;
    for(ssize_t i = 0; i < npoints; ++i) {
        ls.emplace_back(nd);
        for(auto &x: ls.back())
            x = std::ldexp(double(mt()), 64);
    }
    for(const auto &v: ls)
        dci.add_item(v);//, dci2.add_item(v);
    auto topn = dci.query(ls[0], 3);
    std::fprintf(stderr, "topn: %zu is n\n", topn.size());
    std::cerr << "added item to dci\n";
}
