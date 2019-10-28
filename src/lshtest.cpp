#include "frp/lsh.h"

using namespace frp;

int main() {
    MatrixLSHasher<> mat(12, 16);
    std::vector<blaze::DynamicVector<float>> tmp;
    while(tmp.size() < 500) {
        tmp.emplace_back(16);
        randomize(tmp.back());
    }
    for(const auto &v: tmp) {
        std::fprintf(stderr, "hash: %zu\n", mat(v));
    }
    
}
