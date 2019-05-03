#include "include/frp/dci.h"

int main() {
    frp::dci::DCI<blaze::DynamicVector<float>> dc(10, 20, 200);
    blaze::DynamicVector<float> zomg{1,2,3,4,5};
    auto q = dc.query(zomg, 10);
}
