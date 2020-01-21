#ifndef FROOPY_MEX_H__
#define FROOPY_MEX_H__
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include "blaze/Math.h"

namespace frp {
    template<typename FT, bool SO>
    blaze::CompressedMatrix<FT, SO> parse_mm(std::string fn) {
        std::ifstream ifs(fn);
        std::string line;
        do {
            std::getline(ifs, line);
        } while(line.empty() || line.front() == '%');
        char *s = line.data();
        while(std::isspace(*s)) ++s;
        unsigned long nrows = std::strtoul(s, &s, 10);
        do ++s while(std::isspace(*s));
        unsigned long ncols = std::strtoul(s, &s, 10);
        do ++s while(std::isspace(*s));
        unsigned long nnz = std::strtoul(s, nullptr, 10);
        blaze::CompressedMatrix<FT, SO> ret(nrows, ncols);
        ret.reserve(nnz);
        while(std::getline(ifs, line)) {
            s = line.data();
            while(std::isspace(*s)) ++s;
            auto rownum = std::strtoul(s, &s, 10) - 1;
            do ++s while(std::isspace(*s));
            auto colnum = std::strtoul(s, &s, 10) - 1;
            do ++s while(std::isspace(*s));
            double val = std::strtod(s, nullptr);
            ret.insert(rownum, colnum, val);
        }
        return ret;
    }
}

#endif /* FROOPY_MEX_H__ */
