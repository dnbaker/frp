#include "kspp/ks.h"
#include <vector>

int main() {
    std::vector<ks::KString> strings{10};
    for(size_t i(0); i < strings.size(); ++i) {
        if(i & 1) strings[i].putuw_((i << 4) + (i << 2) * i);
        else      strings[i].putsn_("Hello world", 11);
    }
    for(auto &str: strings) {
        str.terminate();
    }
    for(auto &str: strings) {
        std::fprintf(stderr, "Element in string vec: %s\n", str.data());
    }
}
