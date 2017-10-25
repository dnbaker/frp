#ifndef _GFRP_MACH_H__
#define _GFRP_MACH_H__
#include <cassert>
#include <tuple>
#include <unistd.h>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "kspp/ks.h"

namespace gfrp { namespace mach {

void print_toks(std::vector<ks::KString> &strings) {
    ks::KString tmp;
    tmp.sprintf("Num toks: %zu\t", strings.size());
    for(const auto &str: strings) tmp.resize(tmp.size() + str.size());
    for(const auto &str: strings) tmp += str, tmp += ',';
    tmp.pop();
    std::fprintf(stderr, "toks: %s\n", tmp.data());
}

#ifdef __APPLE__
#define CACHE_CMD_STR "/usr/sbin/system_profiler SPHardwareDataType"
#else
#define CACHE_CMD_STR "lscpu"
#endif

template<typename SizeType=size_t>
std::tuple<SizeType, SizeType, SizeType> get_cache_sizes() {
    FILE *fp(popen(CACHE_CMD_STR, "r"));
    char buf[1 << 16]{0}; 
    char *line(nullptr);
    SizeType ret[]{0, 0,0};
    SizeType *ptr(nullptr);
    while((line = std::fgets(buf, sizeof(buf), fp))) {
        if(std::strstr(line, "ache") == nullptr) continue;
        if(std::strstr(line, "L") == nullptr) continue;
        auto toks(ks::toksplit<int>(line, std::strlen(line), 0));
        if(toks[0] == "L1") {
            ptr = ret;
        } else if(toks[0] == "L2") {
            ptr = ret + 1;
        } else if(toks[0] == "L3") {
            ptr = ret + 2;
        } else {
            std::fclose(fp);
            std::fprintf(stderr, "DIE (%s)\n", toks[0].data());
            std::exit(1);
        }
#ifdef __APPLE__
        const auto &endtok(toks.back());
        const auto &magtok(toks[toks.size() - 2]);
        *ptr = std::atoi(magtok.data());
        const char sizechar(endtok[0]);
#elif linux
        const char *tmp(toks.back().data());
        *ptr = std::atoi(tmp);
        while(std::isdigit(*tmp)) ++tmp;
        const char sizechar(*tmp);
#else
#error("Unsupported platform")
#endif
        assert(std::isalpha(sizechar));
        switch(sizechar) {
            case 'T': case 't': *ptr <<= 10;
            case 'G': case 'g': *ptr <<= 10;
            case 'M': case 'm': *ptr <<= 10;
            case 'K': case 'k': *ptr <<= 10;
        }
    }

    std::fclose(fp);
    return std::make_tuple(ret[0], ret[1], ret[2]);
}

}} // namespace gfpr::mach


#endif // #ifndef _GFRP_MACH_H__
