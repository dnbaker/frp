#include "gfrp/gfrp.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std::chrono;

using namespace gfrp::tx;
using namespace gfrp::dist;
using namespace blaze;

template<typename T>
bool has_neg(const T &mat) {
    for(size_t i(0); i < mat.rows(); ++i)
        for(size_t j(0); j < mat.columns(); ++j)
            if(mat(i, j) < 0) return true;
    return false;
}
template<typename T>
bool has_vneg(const T& vec) {for(const auto &el: vec){ if(el < 0) return true;} return false;}

int main(int argc, char *argv[]) {
    const std::size_t size(argc <= 1 ? 1 << 20: std::strtoull(argv[1], 0, 10));
    DynamicMatrix<float, rowMajor> mat(size, size);
    randomize(mat);
    //std::cerr << mat;
    const auto i(system_clock::now()); 
    rademacher(mat);
    DynamicVector<float> vec(size * size);
    randomize(vec);
    rademacher(vec);
    const auto j(system_clock::now()); 
    std::cout << "took " << (duration<double>(j - i).count()) << "s\n";
    auto shuffled_vector(make_shuffled<std::vector<std::size_t>>(size));
    DynamicVector<std::size_t> shufvec(size);
    std::memcpy(&shufvec[0], &shuffled_vector[0], shufvec.size() * sizeof(std::size_t));
    std::cout << shufvec;
    gfrp::linalg::gram_schmidt(mat, true);
    std::cout << mat;
    gaussian_fill(vec);
}
