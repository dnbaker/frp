#include "frp/linalg.h"
#include "frp/dist.h"
using namespace frp; // enter Froopyland!
using namespace frp::linalg;


int main(int argc, char *argv[]) {
    unsigned size(argc > 1 ? std::atoi(argv[1]): 10), nrows(size), ncols(size), flags(ORTHONORMALIZE);
    blaze::DynamicMatrix<FLOAT_TYPE> input(nrows, ncols), ret1, ret2;
    for(size_t i(0); i < input.rows(); ++i) {
        auto r(row(input, i));
        gaussian_fill(r, 1337 * i);
    }
    gram_schmidt(input, ret1, flags);
    qr_gram_schmidt(input, ret2, flags);
    auto ret3(qr_gram_schmidt(input, flags));
    std::cerr << "input:\n" << input;
    std::cerr << "ret1:\n" << ret1;
    std::cerr << "ret2:\n" << ret2;
    std::cerr << "ret3:\n" << ret3 << '\n';
    for(size_t i(0); i < size; ++i) {
        for(size_t j(0); j < size; ++j) {
            std::fprintf(stderr, "Dot products of row %zu with row %zu are %f, %f\n", i, j, dot(row(ret1, i), row(ret1, j)), dot(row(ret2, i), row(ret2, j)));
        }
        std::fprintf(stderr, "The norm of the rows are %lf, %lf, %lf\n", norm(row(ret1, i)), norm(row(ret2, i)), norm(row(ret3, i)));
    }
}
