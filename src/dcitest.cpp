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
    //std::cerr << "Distances! " << dists << '\n';
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
        //std::fprintf(stderr, "Label stuff2nn %zu\n", i);
        auto r = row(mat, i);
        //std::cerr << "Matrix row: " << r;
        auto func = [&r](size_t j, size_t k){return r[j] > r[k];};
        size_t heapsz = 0;
        auto pq(row(ret, i));
        assert(k == pq.size());
        size_t j;
        for(j = 0;j < mat.rows();++j) {
            auto pqp = &pq[0];
            if(heapsz < pq.size()) {
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
    int nd = 40, npoints = 1000, n = 10;
    DCI<blaze::DynamicVector<FLOAT_TYPE>> dci(4, 20, nd, 1e-5, true);
    {
        // make sure it works with < nd
        DCI<blaze::DynamicVector<FLOAT_TYPE>, uint32_t, float, std::deque> dcid(4, 20, nd, 1e-5, false);
        DCI<blaze::DynamicVector<FLOAT_TYPE>> tmp(4, 3, nd, 1e-5, true);
    }
    std::cerr << "made dci\n";
    std::vector<blaze::DynamicVector<FLOAT_TYPE>> ls;
    std::mt19937_64 mt;
    std::normal_distribution<FLOAT_TYPE> gen(0., 1);
    gen.reset();
    for(ssize_t i = 0; i < npoints; ++i) {
        omp_set_num_threads(std::thread::hardware_concurrency());
        ls.emplace_back(nd);
        for(auto &x: ls.back())
            x = gen(mt);
    }
    std::fprintf(stderr, "Generated\n");
    for(const auto &v: ls)
        dci.add_item(v);//, dci2.add_item(v);
    std::fprintf(stderr, "Added\n");
    auto [x, y] = nn_data(dci);
    //std::fprintf(stderr, "nn\n");
    auto nnmat = distmat2nn(x, std::max(n, nd - 15));
    std::priority_queue<frp::dci::ProjID<FLOAT_TYPE, int>> pqs;
    for(int i = 0; i < npoints; ++i) {
        pqs.emplace(norm(ls[0] - ls[i]), i);
        if(pqs.size() > n) {
            //std::fprintf(stderr, "Last thing: %f\n", pqs.top().f());
            pqs.pop();
        }
    }
    while(pqs.size()) {
        //std::fprintf(stderr, "popping %f\n", pqs.top().f());
        pqs.pop();
    }
    //std::cerr << nnmat << '\n';
    auto topn = dci.query(ls[0], n);
    std::fprintf(stderr, "topn, where n is %zu: \n\n", topn.size());
    auto tnbeg = topn.begin();
    assert(tnbeg->id() == 0);
    double mv = norm(ls[tnbeg++->id()] - ls[0]);
    std::fprintf(stderr, "first dist: %le\n", mv);
    assert(mv == 0.0); // Should be itself
    do {
        auto id = tnbeg->id();
        blaze::DynamicVector<FLOAT_TYPE> &rl(ls[id]);
        blaze::DynamicVector<FLOAT_TYPE> &rr(ls[0]);
        double newv = norm(rl - rr);
        assert(mv <= newv);
        std::fprintf(stderr, "dist: %f, id %u\n", newv, unsigned(id));
    } while(++tnbeg != topn.end());
    auto dcid2 = dci.template cvt<std::set>();
}
