#include "include/gfrp/mach.h"
using namespace gfrp::mach;

int main() {
    auto szs(get_cache_sizes());
    std::fprintf(stderr, "L2 cache size: %zu.\n", szs[1]);
    std::fprintf(stderr, "L3 cache size: %zu.\n", szs[2]);
}
