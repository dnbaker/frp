#undef NDEBUG
#include "frp/linalg.h"
#include <iostream>

using namespace frp;
using namespace linalg;

// c = np.dot((X.T - np.mean(X, axis=1)).T, (X.T - np.mean(X, axis=1))) * np.true_divide(1, wsum[0] - 1)
// Python code above works.
// The key is take the matrix, sub
int main(int argc, char *argv[]) {
    int nrows = argc == 1 ? 25: std::atoi(argv[1]), ncols = argc < 3 ? 5: std::atoi(argv[2]), ncomp = argc < 4 ? 3: std::atoi(argv[3]);
    blaze::DynamicMatrix<float> mat(nrows, ncols);
    std::mt19937_64 mt;
    std::uniform_real_distribution<float> gen;
    for(size_t i = 0; i < mat.rows(); ++i) for(size_t j = 0; j < mat.columns(); ++j)
        mat(i, j) = gen(mt);
    auto c = naive_cov(mat);
    auto c2 = naive_cov(mat, false);
    auto s = blaze::sum<blaze::columnwise>(mat);
    std::cout <<" mat \n" << mat << '\n';
    std::cout << "cov \n" << c << '\n';
    std::cout << "sum \n" << s << '\n';
    std::cout << "sample cov \n" << c2 << '\n';
    auto [x, y] = pca(mat, true, true, ncomp);
    std::fprintf(stderr, "Dims of x: %zu/%zu rows/columns\n", x.rows(), x.columns());
    std::fprintf(stderr, "Dims of mat: %zu/%zu rows/columns\n", mat.rows(), mat.columns());
    auto txdata = mat * x;
    std::cout << txdata << '\n';
    std::fprintf(stderr, "Dims of txdata: %zu/%zu rows/columns\n", txdata.rows(), txdata.columns());
    std::fprintf(stderr, "Subsample to 3\n");
}
