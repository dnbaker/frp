#include "x86intrin.h"
#include "immintrin.h"
#include <cmath>
#include "gfrp/util.h"
#include "avx_mathfun.h"

using namespace gfrp;

int main(int argc, char *argv[]) {
    size_t niter(argc == 1 ? (1 << 12): std::atoi(argv[1]));
    blaze::DynamicVector<float> vec1(1 << 16);
    blaze::DynamicVector<float> vec2(1 << 17);
    for(size_t i(0); i < vec1.size(); ++i) {
        vec1[i] = i * i - 7 * i;
    }
    {
        Timer t("naive");
        for(size_t j(niter); j--;) {
            for(size_t i(vec1.size()); i; --i) {
                vec2[(i - 1) << 1] = std::cos(vec1[i - 1]);
                vec2[(i << 1) - 1] = std::sin(vec1[i - 1]);
            }
        }
    }
    std::cerr << subvector(vec2, 0, 10);
    reset(vec2);
    {
        Timer t("direct sin/cos");
        __m256 *dst((__m256 *)&vec2[0]), *src(((__m256 *)&vec1[0]));
        static const __m256 inc(_mm256_set_ps(0.5, 0, 0.5, 0,                
                                              0.5, 0, 0.5, 0));
        __m256 *dstend((__m256 *)&vec2[vec2.size()]);
        const size_t fac(sizeof(__m256) / sizeof(float));
        for(size_t j(niter); j--;) {
            for(size_t i(vec2.size()); i; i -= fac) {
                _mm256_storeu_ps(&vec2.at(i - fac),
                    _mm256_cos_ps(
                        _mm256_add_ps(
                            _mm256_set_ps(
                                vec1[(i >> 1) - 1], vec1[(i >> 1) - 1],
                                vec1[(i >> 1) - 2], vec1[(i >> 1) - 2],
                                vec1[(i >> 1) - 3], vec1[(i >> 1) - 3],
                                vec1[(i >> 1) - 4], vec1[(i >> 1) - 4])
                                      , inc)));
                std::fprintf(stderr, "min value %zu.\n", (i >> 1) - 4);
            }
        }
    }
    std::cerr << subvector(vec2, 0, 10);
}
