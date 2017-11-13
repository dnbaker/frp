#include "gfrp/gfrp.h"
using namespace gfrp;

int main() {
    auto arr(aes::seed_to_array<size_t, 3>(1337));
    for(auto el: arr) std::cerr << "el: " << el << '\n';
}
