#ifndef _GFRP_MACH_H__
#define _GFRP_MACH_H__
#include <cassert>
#include <string>
#include <unistd.h>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "kspp/ks.h"
#include "frp/util.h"

namespace frp { namespace mach {

void print_toks(std::vector<ks::string> &strings) {
    ks::string tmp;
    tmp.sprintf("Num toks: %zu\t", strings.size());
    for(const auto &str: strings) tmp.resize(tmp.size() + str.size());
    for(const auto &str: strings) tmp += str, tmp += ',';
    tmp.pop();
    fprintf(stderr, "toks: %s\n", tmp.data());
}

#ifdef __APPLE__
#define CACHE_CMD_STR "/usr/sbin/system_profiler SPHardwareDataType"
#else
#define CACHE_CMD_STR "lscpu"
#endif

template<typename T>
using ref = T&;

struct CacheSizes {
    size_t l1, l2, l3;
    operator ref<size_t [3]>() {
        return reinterpret_cast<ref<size_t [3]>>(*this);
    }
    CacheSizes(size_t l1a, size_t l2a, size_t l3a): l1(l1a), l2(l2a), l3(l3a) {}
    CacheSizes() {memset(this, 0, sizeof(*this));}
    std::string str() const {
        char buf[64];
        sprintf(buf, "L1:%zu,L2:%zu,L3:%zu", l1, l2, l3);
        return buf;
    }
};

template<typename SizeType=size_t>
CacheSizes get_cache_sizes() {
    FILE *fp(popen(CACHE_CMD_STR, "r"));
    char buf[1 << 16];
    memset(buf, 0, sizeof(buf));
    CacheSizes ret;
    SizeType  *ptr;
    char     *line;
    while((line = fgets(buf, sizeof(buf), fp))) {
        if(strstr(line, "ache") == nullptr) continue;
        if(strstr(line, "L") == nullptr) continue;
        auto toks(ks::toksplit<int>(line, strlen(line), 0));
        if(toks[0] == "L1i") {
            continue;
        } else if(toks[0] == "L1d") {
            ptr = &ret[0];
        } else if(toks[0] == "L2") {
            ptr = &ret[1];
        } else if(toks[0] == "L3") {
            ptr = &ret[2];
        } else {
            fclose(fp);
            fprintf(stderr, "DIE (%s)\n", toks[0].data());
            exit(1);
        }
#ifdef __APPLE__
        const auto &endtok(toks.back());
        const auto &magtok(toks[toks.size() - 2]);
        *ptr = atoi(magtok.data());
        const char sizechar(endtok[0]);
#else
        const char *tmp(toks.back().data());
        *ptr = atoi(tmp);
        while(isdigit(*tmp)) ++tmp;
        const char sizechar(*tmp);
#endif
        assert(isalpha(sizechar));
        switch(sizechar) {
            case 'T': case 't': *ptr <<= 40; break;
            case 'G': case 'g': *ptr <<= 30; break;
            case 'M': case 'm': *ptr <<= 20; break;
            case 'K': case 'k': *ptr <<= 10; break;
        }
    }

    fclose(fp);
    return ret;
}

}} // namespace gfpr::mach


#endif // #ifndef _GFRP_MACH_H__
