#include "include/frp/dci.h"
#include <iostream>

using namespace frp;
using namespace dci;
int main() {
    DCI<blaze::DynamicVector<float>> dci(100, 4, 10);
    std::cerr << "made dci\n";
    blaze::DynamicVector<float> zomg(10);
    randomize(zomg);
    dci.add_item(zomg);
    dci.query(zomg, 3);
    std::cerr << "added item to dci\n";
}
