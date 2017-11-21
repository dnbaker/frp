#include "kspp/ks.h"
#include <vector>

float randf() {
    return (float)rand() / RAND_MAX;
}

int main() {
    std::vector<ks::KString> strings{10};
    for(size_t i(0); i < strings.size(); ++i) {
        if(i & 1) strings[i].putuw_((i << 4) + (i << 2) * i);
        else      strings[i].putsn_("Hello world\t", 12), strings[i].putuw_((i << 5) + (((i << 3) ^ i) ^ (i << 16))) ;
    }
    std::vector<float> floats;
    while(floats.size() < 50) floats.push_back(randf());
    for(auto &str: strings) {
        str.terminate();
    }
    for(auto &str: strings) {
        std::fprintf(stderr, "Element in string vec: %s\n", str.data());
    }
    ks::string tmp(1 << 10);
    for(const auto el: floats) {
        tmp.sprintf(",%e", el);
    }
    std::fprintf(stderr, "tmp: '%s'\n");
    auto str(strings[0].str());
    std::fprintf(stderr, "String copy of el 1: %s\n", str.data());
}
