#ifndef FRP_DCI_H__
#define FRP_DCI_H__
#include "blaze/Math.h"
#include <map>
#include <cmath>
#include <set>
#include "omp.h"

namespace frp {

namespace dci {

template<typename FloatType, bool SO, typename T>
auto dot(const blaze::DynamicVector<FloatType, SO> &r, const T &x) {
    return blaze::dot(r, x);
}

template<typename ValueType, typename IdType=std::uint32_t, typename FType=float, template <typename...> typename MapType=std::map>
class DCI {
    /*
     To use this: Hold values in their own container. (This is non-owning.)
     Provide a distance metric/dot function which performs dot through ADL.
     https://arxiv.org/abs/1512.00442
    */
    using map_type = MapType<FType, IdType>;
    using matrix_type = blaze::DynamicMatrix<FType>;
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
    bool should_stop(size_t i, const std::set<IdType> &x, unsigned k) {
        return x.size() >= k; // Arbitrary, probably bad. (Will implement better later.)
    }
    void query(const ValueType &val, unsigned k) {
        //std::vector<IdType> lbs, ubs;
        std::vector<std::set<IdType>> candidatesvec(l_);
        std::vector<std::vector<unsigned>> countsvec(l_);
        for(auto &v: countsvec) v.resize(size());
        for(size_t i = 0; i < size(); ++i) {
            for(size_t l = 0; l < l_; ++l) {
                auto &candidates = candidatesvec[l];
                auto &C = countsvec[l];
                throw std::runtime_error("NotImplemented: getting ith closest along line.");
#if 0
                for(size_t  j = 0; j < m_; ++j) {
                    auto index = ind(j, l);
                    assert(index < mat_.rows());
                    auto &map = map_[index];
                    auto r = row(mat_, index);
                    FType dist = dot(val, r);
                    auto it1 = map.lower_bound(dist), it2 = map.upper_bound(dist);
                    IdType id;
                    if(it1->first == dist)
                        id = it1->second;
                    else if(it2->first == dist) id = it2->second;
                    else {
                        id = blaze::sqrNorm(val - (*val_ptrs_[it1->second])) < blaze::sqrNorm(val - (*val_ptrs_[it2->second]))
                             ? it1->second: it2->second;
                    }
                    ++C[id];
                }
                for(size_t  j = 0; j < m_; ++j) {
                    if(C[j] == m_)  // Should this be >= instead of ==, like the paper says?
                        candidates.insert(id);
                }
#endif
                if(should_stop(i, candidates, k)) break;
            }
        }
        throw std::runtime_error("NotImplemented: final collation of the other lists of items.");
    }
    size_t size() const {return n_inserted_;}
    size_t ind(size_t m, size_t l) const {
        return l * m_ + m;
    }
    std::pair<size_t, size_t> offset2ind(size_t offset) const {return std::pair<size_t, size_t>(offset % m_, offset / m_);}
};

}

} // namespace frp

#endif
