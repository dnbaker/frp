#include "boost/math/special_functions/gamma.hpp"
#include <cstdio>

int main() {
    double input(.1);
    double input2(256);
    float finput(.1);
    float finput2(256);
    std::fprintf(stderr, "gammaq [a = 0.1, q = 256: %lf]\n", boost::math::gamma_q_inv(input2, input));
    std::fprintf(stderr, "gammap [a = 0.1, p = 256: %lf]\n", boost::math::gamma_p_inv(input2, input));
    std::fprintf(stderr, "gammap [a = 0.1, p = 256: %lf]\n", boost::math::gamma_p_inv(finput2, finput));
}
