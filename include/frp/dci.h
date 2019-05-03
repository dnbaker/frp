#ifndef FRP_DCI_H__
#define FRP_DCI_H__
#ifndef FHT_HEADER_ONLY
#define FHT_HEADER_ONLY 1
#endif
#include "FFHT/fast_copy.h"
#include "blaze/Math.h"
#include <map>
#include <cmath>
#include <set>
#include "omp.h"
#include "./heap.h"

namespace frp {

namespace dci {

template<typename FloatType, bool SO, typename T>
auto dot(const blaze::DynamicVector<FloatType, SO> &r, const T &x) {
    return blaze::dot(r, x);
}

template<typename ValueType,
         typename IdType=std::uint32_t, typename FType=float,
         template <typename...> class MapType=std::map,
         template <typename...> class SetTemplate=std::set>
class DCI {
    /*
     To use this: Hold values in their own container. (This is non-owning.)
     Provide a distance metric/dot function which performs dot through ADL.
     https://arxiv.org/abs/1512.00442
    */
    using map_type = MapType<FType, IdType>;
    using set_type = SetTemplate<IdType>;
    using matrix_type = blaze::DynamicMatrix<FType>;
    using bin_tree_iterator = typename set_type::const_iterator;
    matrix_type mat_;
    std::vector<map_type> map_;
    std::vector<const ValueType*> val_ptrs_;
    size_t m_, l_;
    size_t n_inserted_;
    double eps_;
public:
    size_t total() const {return m_ * l_;}
    DCI(size_t m, size_t l, size_t d, double eps=1e-5, bool orthonormalize=false): m_(m), l_(l), mat_(m * l, d), map_(m * l), n_inserted_(0), eps_(eps) {
        blaze::randomize(mat_);
        std::fprintf(stderr, "Made mat of %zu/%zu\n", mat_.rows(), mat_.columns());
        if(orthonormalize) {
            std::fprintf(stderr, "about to qr\n");
            try {
                matrix_type q, r;
                blaze::qr(mat_, q, r);
                std::fprintf(stderr, "coldist %lf, rowdists %lf, q sizes %zu/%zu\n", dot(column(r, 0), column(r, 1)),  dot(row(r, 0), row(r, 1)), q.rows(), q.columns());
                std::fprintf(stderr, "coldist %lf, rowdists %lf, q sizes %zu/%zu\n", dot(column(q, 0), column(q, 1)),  dot(row(q, 0), row(q, 1)), q.rows(), q.columns());
                assert(mat_.columns() == q.columns());
                assert(mat_.rows() == q.rows());
                swap(mat_, q);
            } catch(const std::exception &ex) { // Orthonormalize
                std::fprintf(stderr, "failure: %s\n", ex.what()); throw;
            }
        }
        for(size_t i = 0; i < mat_.rows(); ++i) {
            auto r = blaze::row(mat_, i);
            r *= 1./ norm(r);
        }
    }
    void add_item(const ValueType &val) {
        IdType ind = n_inserted_++;
        for(size_t i = 0; i < mat_.rows(); ++i) {
            auto &map = map_[i];
            FType key = dot(val, row(mat_, i));
            auto it = map.find(key);
            while(it != map.end() && key != 0.) {
                std::fprintf(stderr, "Warning: dot product between val and our row is the same as another's and is nonzero, which is unexpected. This might point to duplicates\n");
                key = std::nextafter(key, std::numeric_limits<FType>::max());
                it = map.find(key);
            }
            map.emplace(key, ind);
            val_ptrs_.emplace_back(std::addressof(val));
            assert(val_ptrs_.size() == n_inserted_);
        }
    }
    bool should_stop(size_t i, const std::set<IdType> &x, unsigned k) const {
        return x.size() >= k; // Arbitrary, probably bad. (Will implement better later.)
    }
    static auto next_best(const map_type &map, std::pair<bin_tree_iterator, bin_tree_iterator> &bi, FType val) {
        if(bi.first != map.begin()) {
            if(bi.second != map.end()) {
                auto diff = std::abs(bi.first->first - val) - std::abs(bi.second->first - val);
                if(diff < 0.)
                    return &*(bi.first--);
                else
                    return &*(bi.second--);
            }
            return &*(bi.first--);
        }
        if(bi.second != map.end())
            return &*(bi.second--);
        return nullptr;
    }
    std::vector<IdType> query(const ValueType &val, unsigned k) const {

        // First step: dot product the query with all reference positions
        std::vector<std::pair<bin_tree_iterator, bin_tree_iterator>> bounds;
        std::vector<set_type> candidatesvec(l_);
        std::vector<FType> dists(l_ * m_);
        // Get a pair of iterators
        for(size_t i = 0; i < l_ * m_; ++i) {
            FType dist = dot(row(mat_, i), val);
            dists[i] = dists;
            bounds.emplace_back(std::make_pair(map_.lower_bound(dist), map_.upper_bound(dist)));
        }


        // FIXME: consider sparse representation. I don't expect dense to be best.
        // Allocate counts
        std::vector<std::vector<unsigned>> countsvec(l_);
        for(auto &v: countsvec) v.resize(size());

        // Iterate through ith closest along each projection direction.
        for(size_t i = 0; i < size(); ++i) {
            for(size_t l = 0; l < l_; ++l) {
                auto &candidates = candidatesvec[l];
                auto &C = countsvec[l];
                /* 1. Get `ith` closest to q_{jl} [the `dist` above]
                 * 2. 
                 */
                for(size_t j = 0; j < m_; ++j) {
                    auto index = ind(j, l);
                    auto pair = next_best(map_[index], bounds[index], dists[index]);
                    if(!pair) throw std::runtime_error("Failure in navigating tree");
                    if(++countsvec[pair->second] == m_)
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
        size_t di = 0;
        heap::ObjScoreHeap<size_t, std::hash<size_t>, FType> osh;
        for(auto e: u) {
            osh.addh(e, blaze::sqrNorm(*val_ptrs_[e], val));
        }
        std::vector<IdType> ids; ids.reserve(k);
        for(const auto &x: osh)
            ids.push_back(x.first);
        return ids;
    }
    size_t size() const {return n_inserted_;}
    size_t ind(size_t m, size_t l) const {
        return l * m_ + m;
    }
    std::pair<uint32_t, uint32_t> invind(size_t index) const {
        return std::make_pair(uint32_t(index / m_), uint32_t(index % m_));
    }
    std::pair<size_t, size_t> offset2ind(size_t offset) const {return std::pair<size_t, size_t>(offset % m_, offset / m_);}
};

}

} // namespace frp

#endif
