#include "include/frp/dci.h"
#include <iostream>
#include <thread>
#include "omp.h"

using namespace frp;
using namespace dci;

// using UDMF = blaze::StrictlyUpperMatrix<blaze::DynamicMatrix<float>>;
// using UDMU = blaze::StrictlyUpperMatrix<blaze::DynamicMatrix<unsigned>>;
using UDMF = blaze::DynamicMatrix<FLOAT_TYPE>;
using UDMU = blaze::DynamicMatrix<unsigned>;


//StrictlyUpperMatrix

template<typename DCIType>
std::pair<UDMF,UDMU> nn_data(const DCIType &dc) {
    auto a = dc.begin();
    ssize_t dist = std::distance(a, dc.end());
    size_t n = dist;
    UDMF dists(dist, dist);
    _Pragma("omp parallel for")
    for(size_t i = 0; i < n; ++i) {
        const blaze::DynamicVector<FLOAT_TYPE> &vp1 = *dc[i];
        auto r1 = row(dists, i);
        for(size_t j = i + 1; j < n; ++j) {
            const blaze::DynamicVector<FLOAT_TYPE> &vp2 = *dc[j];
            auto d = blaze::norm(vp1 - vp2);
            r1[j] = d;
        }
    }
    UDMU labels(dist, dist);
    std::fprintf(stderr, "Label stuff\n");
    _Pragma("omp parallel for")
    for(size_t i = 0; i < n; ++i) {
        auto r = row(labels, i);
        auto mr = row(dists, i);
        std::iota(r.begin(), r.end(), 0u);
        sort(r.begin(), r.end(), [&mr](auto x, auto y) {return mr[x] < mr[y];});
#if 0
        for(auto x: r) {
            assert(x < mr.size());
            std::fprintf(stderr, "n = %zu speaking now with r %d/%f\n", i, x, float(mr[x]));
        }
#endif
    }
    std::cerr << "Distances! " << dists << '\n';
    std::fprintf(stderr, "Return pair stuff\n");
    return std::make_pair(std::move(dists), std::move(labels));
}

template<typename T1, typename I=std::uint32_t>
auto distmat2nn(const T1 &mat, size_t k) {
    if(mat.columns() > std::numeric_limits<I>::max())
        throw std::runtime_error("Overflow: mat size too large");
    k = std::min(mat.columns(), k);
    blaze::DynamicMatrix<I> ret(mat.columns(), k);
    //#pragma omp parallel for
    for(size_t i = 0; i < mat.rows(); ++i) {
        
        std::fprintf(stderr, "Label stuff2nn %zu\n", i);
        auto r = row(mat, i);
        std::cerr << "Matrix row: " << r;
        auto func = [&r](size_t j, size_t k){return r[j] > r[k];};
        size_t heapsz = 0;
        auto pq(row(ret, i));
        assert(k == pq.size());
        size_t j;
        for(j = 0;j < mat.rows();++j) {
            std::fprintf(stderr, "%zu\n", j);
            auto pqp = &pq[0];
            if(heapsz < pq.size()) {
#if !NDEBUG
                size_t oldsz = heapsz;
#endif
                pq[heapsz] = j;
                if(++heapsz == pq.size())
                    std::make_heap(pqp, pqp + heapsz, func);
            } else if(func(j, pq[0])) {
                assert(pq.size() >= heapsz);
                std::pop_heap(pqp, pqp + heapsz, func);
                pq[heapsz - 1] = j;
                std::push_heap(pqp, pqp + heapsz, func);
            }
        }
        for(auto it = pq.end();it != pq.begin();std::pop_heap(pq.begin(), it--, func));
        assert(std::is_sorted(pq.begin(), pq.end(), func));
        std::cerr << pq << '\n';
#if 0
        for(auto v: pq) {
            if(r[v] > r[pq[0]]) {
                std::fprintf(stderr, "WOOOrv: %e. rpq: %e\n", r[v], r[pq[0]]);
            }
            else {
                std::fprintf(stderr, "NOOOrv: %e. rpq: %e\n", r[v], r[pq[0]]);
            }
        }
#endif
    }
    return ret;
}


int main() {
    int nd = 40;
    DCI<blaze::DynamicVector<FLOAT_TYPE>> dci(20, 10, nd);
    //DCI<blaze::DynamicVector<float>> dci2(10, 4, nd, 1e-5, true);
    std::cerr << "made dci\n";
    std::vector<blaze::DynamicVector<FLOAT_TYPE>> ls;
    std::mt19937_64 mt;
    std::normal_distribution<FLOAT_TYPE> gen;
    omp_set_num_threads(std::thread::hardware_concurrency());
    for(size_t i = 0; i < 100; ++i) {
        ls.emplace_back(nd);
        for(auto &x: ls.back())
            x = gen(mt);
        std::cerr << ls.back() << '\n';
    }
    for(const auto &v: ls)
        dci.add_item(v);//, dci2.add_item(v);
    auto [x, y] = nn_data(dci);
    auto nnmat = distmat2nn(x, std::max(3, nd - 15));
    dci.query(ls[0], 3);
    std::cerr << "added item to dci\n";
}
