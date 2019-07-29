#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>


template<typename T, typename T1=T>
void perform_fht(T *input, size_t l2, bool renorm=true, T1 U1=1., T1 U2=1., T1 V1=1., T1 V2=-1.) {
    auto n = size_t(1) << l2;
    for(size_t i = 0; i < l2; ++i) {
        size_t s1 = size_t(1) << i, s2 = s1 << 1;
        for(size_t j = 0; j < n; j += s2) {
            for(size_t k = 0; k < s1; ++k) {
                auto u = input[j + k], v = input[j + k + s1];
                input[j + k] = u * U1 + V1 * v;
                input[j + k + s1] = u * U2 + V2 * v;
            }
        }
    }
    if(renorm)  std::for_each(input, input + n, [mult=1. / std::pow(std::sqrt(2.), l2)](auto &x) {x *= mult;});
}

template<typename T>
void randomize(T &x, uint64_t seed=0) {
    seed = seed ? seed: std::pow(x.size(), x.size());
    std::mt19937_64 mt(seed);
    std::normal_distribution<double> gen;
    for(auto &e: x)
        e = gen(mt);
}

int main() {
    int n = 14;
    std::vector<float> v(1 << n);
    for(auto &i: v) i = std::sqrt(double(std::rand()) / RAND_MAX);
    std::mt19937_64 mt;
    for(auto &i: v) {
        float tmp;
        std::srand(*(int *)&tmp);
        mt.seed(std::rand());
        i = mt() / double(std::numeric_limits<typename std::mt19937_64::result_type>::max());
    }
#define show(v) \
    for(size_t i = 0; i < n; std::fprintf(stderr, "%f,", v[i++])); std::fputc('\n', stderr)
    show(v);
    printf("Rotating 90 degrees each time [period of 2]\n");
    perform_fht(v.data(), n);
    perform_fht(v.data(), n);
    show(v);
    printf("Rotating 90 degrees each time [period of 4]\n");
    for(int i = 0; i < 24; ++i) {
        perform_fht(v.data(), n, true, 1., 1., -1., 1.);
        show(v);
    }
    printf("Rotating 60 degrees each time [period of 6]\n");
    for(int i = 0; i < 24; ++i) {
        double p1 = 0.8660254037844386;
        perform_fht(v.data(), n, false, p1, .5, -.5, p1); // Periodicity of 6
        show(v);
    }
    printf("Rotating 120 degrees each time [period of 3]\n");
    for(int i = 0; i < 21; ++i) {
        double p1 = 0.8660254037844386;
        perform_fht(v.data(), n, false, .5, p1, -p1, .5); // Periodicity of 3
        show(v);
    }
    printf("Rotating ??? [period of 14]\n");
    auto v1 = M_PI / 3.5, c1 = std::cos(v1), s1 = std::sin(v1); // 14
    for(int i = 0; i < 28; ++i) {
        std::fprintf(stderr, "iteration %d\n", i);
        show(v);
        perform_fht(v.data(), n, false, s1, c1, -c1, s1); // Periodicity of 6
    }
    std::fprintf(stderr, "iteration done\n");
    printf("Rotating ??? [period of 7]\n");
    for(int i = 0; i < 28; ++i) {
        std::fprintf(stderr, "iteration %d\n", i);
        show(v);
        perform_fht(v.data(), n, false, c1, s1, -s1, c1); // Periodicity of 7
    }
    std::fprintf(stderr, "iteration done\n");
    show(v);
    printf("Rotating ??? [period of ???]\n");
    for(int i = 0; i < 20; ++i) {
        std::fprintf(stderr, "iteration %d\n", i);
        show(v);
        perform_fht(v.data(), n, false, c1, s1, s1, -c1); // Periodicity of 7
    }
    std::fprintf(stderr, "iteration done\n");
    std::fprintf(stderr, "RANDOMIZE\n");
    randomize(v);
    show(v);
    printf("Rotating ??? (the last one but backwards ish?)[period of ???]\n");
    for(int i = 0; i < 20; ++i) {
        std::fprintf(stderr, "iteration %d\n", i);
        show(v);
        perform_fht(v.data(), n, false, s1, c1, c1, -s1); // Periodicity of 7
    }
    std::fprintf(stderr, "iteration done\n");
    show(v);
#if 0
    printf("Rotating ??? [period of ???] two negatives\n");
    for(int i = 0; i < 20; ++i) {
        std::fprintf(stderr, "iteration %d\n", i);
        show(v);
        perform_fht(v.data(), n, false, -c1, s1, c1, -s1); // Periodicity of 7
    }
    std::fprintf(stderr, "iteration done\n");
    show(v);
#endif
//perform_fht(v.data(), n, true, 1., 1., -1., 1.);
//perform_fht(v.data(), n, true, 1., 1., -1., 1.);
//perform_fht(v.data(), n, true, 1., 1., -1., 1.);
//perform_fht(v.data(), n, true, 1., 1., -1., 1.);
}
