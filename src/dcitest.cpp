#include "include/frp/dci.h"
#include <iostream>
#include <thread>
#include "omp.h"

using namespace frp;
using namespace dci;

// using UDMF = blaze::StrictlyUpperMatrix<blaze::DynamicMatrix<float>>;
// using UDMU = blaze::StrictlyUpperMatrix<blaze::DynamicMatrix<unsigned>>;
using UDMF = blaze::DynamicMatrix<float>;
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
        const blaze::DynamicVector<float> &vp1 = *dc[i];
        auto r1 = row(dists, i);
        for(size_t j = i + 1; j < n; ++j) {
            const blaze::DynamicVector<float> &vp2 = *dc[j];
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
        auto func = [&r](size_t j, size_t k){return r[j] < r[k];};
        size_t heapsz = 0;
        auto pq(row(ret, i));
        assert(k == pq.size());
        size_t j;
        for(j = 0;j < mat.rows();++j) {
            auto pqp = &pq[0];
            if(heapsz < pq.size()) {
                //std::fprintf(stderr, "p[heapsz]: %zu\n", j);
                pq[heapsz] = j;
                std::push_heap(pqp, pqp + heapsz, func);
                ++heapsz;
            } else if(func(pq[0], j)) {
                //std::fprintf(stderr, "whoa nelly: %zu\n", j);
                assert(pq.size() >= heapsz);
                std::pop_heap(pqp, pqp + heapsz - 1, func);
                pq[heapsz - 1] = j;
                std::push_heap(pqp, pqp + heapsz - 1, func);
            }
        }
        for(auto it = pq.end();it-- != pq.begin();std::pop_heap(pq.begin(), it, func));
        for(auto _p: pq) {
            std::fprintf(stderr, "%d/%f\n", _p, r[_p]);
        }
        assert(std::is_sorted(pq.begin(), pq.end(), func));
        for(auto v: pq) {
            if(r[v] > r[pq[0]]) {
                std::fprintf(stderr, "WOOOrv: %f. rpq: %f\n", r[v], r[pq[0]]);
            }
            else {
                std::fprintf(stderr, "NOOOrv: %f. rpq: %f\n", r[v], r[pq[0]]);
            }
        }
    }
    return ret;
}


int main() {
    int nd = 40;
    DCI<blaze::DynamicVector<float>> dci(20, 10, nd);
    //DCI<blaze::DynamicVector<float>> dci2(10, 4, nd, 1e-5, true);
    std::cerr << "made dci\n";
    std::vector<blaze::DynamicVector<float>> ls;
    std::mt19937_64 mt;
    std::normal_distribution<float> gen;
    omp_set_num_threads(std::thread::hardware_concurrency());
    for(size_t i = 0; i < 100; ++i) {
        ls.emplace_back(nd);
        for(auto &x: ls.back())
            x = gen(mt);
    }
    for(const auto &v: ls)
        dci.add_item(v);//, dci2.add_item(v);
    auto [x, y] = nn_data(dci);
    auto nnmat = distmat2nn(x, std::max(3, nd - 15));
    dci.query(ls[0], 3);
    std::cerr << "added item to dci\n";
}
