#include "kspp/ks.h"
#include <vector>

float randf() {
    return (float)rand() / RAND_MAX;
}

int main() {
    using ks::string;
    string z("ZOMG");
    string z2(256);
    while(z2.size() < 16) z2 += 'x';
    z2.terminate();
    std::fprintf(stderr, "str1: %s. str2: %s\n", z.data(), z2.data());
    ks::string other("I am this str");
    std::fprintf(stderr, "Does this str (%s) end in str? %s\n", other.data(), other.endswith("str") ? "true": "false");
    std::fprintf(stderr, "Locate: %s\n", other.locate("str"));
    std::fprintf(stderr, "Locate: %s\n", other.bmlocate("str"));
}
