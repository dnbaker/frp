#ifndef FRP_DCI_H__
#define FRP_DCI_H__
#ifndef FHT_HEADER_ONLY
#define FHT_HEADER_ONLY 1
#endif
#include "./ifc.h"
#include "./util.h"
#include "sdq.h"
#include <deque>
#include "blaze/Math.h"
#include <map>
#include <cmath>
#include <set>
#include <queue>
#include <vector>


namespace frp {

namespace dci {

template<typename FloatType, bool SO, typename T>
auto dot(const blaze::DynamicVector<FloatType, SO> &r, const T &x) {
    return blaze::dot(r, x);
}

template<typename FType, typename SizeType>
struct ProjID: public std::pair<FType, SizeType> {
    template<typename...Args>
    ProjID(Args &&...args): std::pair<FType, SizeType>(std::forward<Args>(args)...) {}
    FType f() const {return this->first;}
    FType id() const {return this->second;}
};

template<typename T>
double cossim(const T &x, const T &y) {
    auto sim = dot(x, y);
    auto xs = dot(x, x), ys = dot(x, y);
    return sim / std::sqrt(xs + ys);
}

template<typename ValueType,
         typename IdType=std::uint32_t, typename FType=float,
         template <typename...> class SetTemplate=std::set>
class DCI {
    /*
     To use this: Hold values in their own container. (This is non-owning.)
     Provide a distance metric/dot function which performs dot through ADL.
     https://arxiv.org/abs/1512.00442
    */
    using ProjI = ProjID<FType, IdType>;
    using map_type = sorted::vector<ProjI>;
    using set_type = SetTemplate<IdType>;
    using matrix_type = blaze::DynamicMatrix<FType>;
    using value_type = ValueType;
    using bin_tree_iterator = typename map_type::const_iterator;
    matrix_type mat_;
    std::vector<map_type> map_;
    std::vector<const value_type*> val_ptrs_;
    size_t m_, l_;
    size_t n_inserted_;
    double eps_;
public:
    size_t total() const {return m_ * l_;}
    DCI(size_t m, size_t l, size_t d, double eps=1e-5, bool orthonormalize=false): m_(m), l_(l), mat_(m * l, d), map_(m * l), n_inserted_(0), eps_(eps) {
        blaze::randomize(mat_);
        std::fprintf(stderr, "Made mat of %zu/%zu\n", mat_.rows(), mat_.columns());
        if(orthonormalize) {
            try {
                matrix_type r, q;
                blaze::qr(mat_, q, r);
                std::fprintf(stderr, "q size: %zu/%zu\n", q.rows(), q.columns());
                std::fprintf(stderr, "mat_ size: %zu/%zu\n", mat_.rows(), mat_.columns());
                assert(dot(column(q, 0), column(q, 1)) < 1e-6);
                assert(mat_.columns() == q.columns());
                assert(mat_.rows() == q.rows());
#if 0
                std::cerr << "q: " << q;
                std::cerr << "r: " << r;    
                auto nonz = [](auto &x) {double ret = 0.; for(size_t i = 0; i < x.rows(); ++i) {for(size_t j = 0; j < x.columns(); ++j) {ret += x(i, j) != 0.;}} return ret;};
                std::cerr << nonz(q) << " q " << nonz(r) << '\n';
#endif
                swap(mat_, q);
                for(size_t i = 0; i < mat_.rows(); ++i) {
                    auto r = blaze::row(mat_, i);
                    r *= 1./ norm(r);
                }
            } catch(const std::exception &ex) { // Orthonormalize
                std::fprintf(stderr, "failure: %s\n", ex.what());
                throw;
            }
        } else { // TODO: consider a triple spinner for generating these random matrix vector multiplies
            for(size_t i = 0; i < mat_.rows(); ++i) {
                auto r = blaze::row(mat_, i);
                r *= 1./ norm(r);
            }
        }
    }
    template<typename I>
    void insert(I i1, I i2) {
        while(i1 != i2)
            add_item(*i1++);
    }
    void add_item(const ValueType &val) {
        IdType ind = n_inserted_++;
        auto tmp = mat_ * val;
        for(size_t i = 0; i < mat_.rows(); ++i) {
            map_[i].emplace(tmp[i], ind);
        }
        val_ptrs_.emplace_back(std::addressof(val));
#if 1
        std::fprintf(stderr, "ind: %u. inserted: %u. valp sz: %zu\n", ind, unsigned(n_inserted_), val_ptrs_.size());
#endif
        assert(val_ptrs_.size() == n_inserted_);
    }
    bool should_stop(size_t i, const set_type &x, unsigned k) const {
        std::fprintf(stderr, "Warning: this needs to be rigorously decided. This code is a simple stopgap measure.\n");
        return x.size() >= k; // Arbitrary, probably bad. (Will implement better later.)
    }
    static const ProjI *next_best(const map_type &map, std::pair<bin_tree_iterator, bin_tree_iterator> &bi, FType val) {
        if(bi.first != map.begin()) {
            if(bi.second != map.end()) {
                auto it = std::abs(bi.first->first - val) > std::abs(bi.second->first - val)
                    ? bi.first-- : bi.second++;
                return &*it;
            }
            auto it = bi.first--;
            return &*it;
        } else if(bi.second != map.end()) {
            auto it = bi.second++;
            return &*it;
        }
        return nullptr;
    }
    std::vector<ProjI> query(const ValueType &val, unsigned k) const {
        bool klt = k < val_ptrs_.size();
        std::vector<ProjI> vs(klt ? k: unsigned(val_ptrs_.size()));
        if(!klt) {
            k = val_ptrs_.size();
            auto prod = mat_ * val;
#if 1
            std::priority_queue<ProjI, std::vector<ProjI>> pq;
            for(size_t i = 0; i < prod.size(); ++i) {
                FType tmp = prod[i];
                if(pq.size() < k) {
                    pq.push(ProjI(tmp, i));
                } else if(pq.size() == k && tmp < pq.top().second) {
                    pq.pop();
                    pq.push(ProjI(tmp, i));
                }
            }
            for(int i = k; i--;pq.pop()) vs[i] = pq.top();
#else
            size_t ind = 0;
            auto it = val_ptrs_.begin();
            std::generate_n(vs.begin(), vs.size(), [&](){return ProjI(blaze::norm(*(*it++) - val), ind++);});
            sort(vs.begin(), vs.end());
#endif
            return vs;
        }

        // First step: dot product the query with all reference positions
        std::vector<std::pair<bin_tree_iterator, bin_tree_iterator>> bounds(l_ * m_);
        std::vector<set_type> candidatesvec(l_);
        std::vector<FType> dists(l_ * m_);
        // Get a pair of iterators
        for(size_t i = 0; i < l_ * m_; ++i) {
            const FType dist = dot(row(mat_, i), val);
            dists[i] = dist;
            const map_type &pos = map_[i];
            static_assert(std::is_same<bin_tree_iterator, typename map_type::const_iterator>::value, "must be");
            static_assert(std::is_same<typename std::remove_const<bin_tree_iterator>::type, std::decay_t<decltype(pos.begin())>>::value, "ZOMG");
            //TD<std::decay_t<decltype(pos.begin())>> td;
            bin_tree_iterator it = std::lower_bound(pos.begin(), pos.end(), dist, [](const auto &x, auto y) {return x.f() < y;});
            bounds[i] = std::make_pair(it, it);
        }


        // FIXME: consider sparse representation. I don't expect dense to be best.
        // Allocate counts
        blaze::DynamicMatrix<uint32_t> countsvec(l_, size());

        // Iterate through ith closest along each projection direction.
        for(size_t i = 0; i < size(); ++i) {
            for(size_t l = 0; l < l_; ++l) {
                auto &candidates = candidatesvec[l];
                auto C = row(countsvec, l);
                /* 1. Get `ith` closest to q_{jl} [the `dist` above]
                 * 2.
                 */
                for(size_t j = 0; j < m_; ++j) {
                    auto index = ind(j, l);
                    auto pair = next_best(map_[index], bounds[index], dists[index]);
                    //TD<decltype(pair)> td;
                    if(!pair) throw std::runtime_error("Failure in navigating tree");
                    if(++C[pair->second] == m_)
                        candidates.insert(j);
                }
                if(should_stop(i, candidates, k)) break;
            }
        }
        auto sit = candidatesvec.begin();
        set_type u = std::move(*sit++);
        while(sit != candidatesvec.end())
            u.insert(sit->begin(), sit->end()), ++sit;
        dists.resize(u.size());
        std::fprintf(stderr, "Begin:\n");
        std::priority_queue<ProjI> pq;
        for(auto it = u.begin(); it != u.end(); ++it) {
            FType tmp = blaze::norm(*val_ptrs_[*it] - val);
            if(pq.size() < k)
                pq.push(ProjI(tmp, *it));
            else if(tmp < pq.top().first) {
                pq.pop();
                pq.push(ProjI(tmp, *it));
            }
        }
        for(size_t i = k; i--; vs[i]= pq.top(), pq.pop());
        return vs;
    }
    size_t size() const {return n_inserted_;}
    size_t ind(size_t m, size_t l) const {
        return l * m_ + m;
    }
    std::pair<uint32_t, uint32_t> invind(size_t index) const {
        return std::make_pair(uint32_t(index / m_), uint32_t(index % m_));
    }
    std::pair<size_t, size_t> offset2ind(size_t offset) const {return std::pair<size_t, size_t>(offset % m_, offset / m_);}
    const value_type *operator[](size_t index) const {return val_ptrs_[index];}
    value_type *operator[](size_t index) {return val_ptrs_[index];}
    auto begin() {return val_ptrs_.begin();}
    auto end() {return val_ptrs_.end();}
    auto begin() const {return val_ptrs_.begin();}
    auto end()   const {return val_ptrs_.end();}
};

}

} // namespace frp

#endif
