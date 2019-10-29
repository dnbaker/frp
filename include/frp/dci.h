#ifndef FRP_DCI_H__
#define FRP_DCI_H__
#ifndef FHT_HEADER_ONLY
#  define FHT_HEADER_ONLY 1
#endif
#include <cstdint>
#include <cstdlib>
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
#include "flat_hash_map/flat_hash_map.hpp"

namespace frp {

namespace dci {

template<typename FloatType, bool SO, typename T>
auto dot(const blaze::DynamicVector<FloatType, SO> &r, const T &x) {
    return blaze::dot(r, x);
}


template<typename T>
struct has_lower_bound_mf_helper
{
    template<typename U, size_t (U::*)() const> struct SFINAE {};
    template<typename U> static char test_fn(SFINAE<U, &U::lower_bound>*);
    template<typename U> static int test_fn(...);
    static constexpr bool value = sizeof(test_fn<T>(nullptr)) == sizeof(char);
};
template<typename T>
struct has_lower_bound_mf: public std::integral_constant<bool, has_lower_bound_mf_helper<T>::value> {};

template<typename T, typename ItemType>
INLINE auto perform_lbound(const T &x, ItemType item, std::false_type ft) {
    return std::lower_bound(std::begin(x), std::end(x), item);
}

template<typename T, typename ItemType>
INLINE auto perform_lbound(const T &x, ItemType item, std::true_type tt) {
    return x.lower_bound(item);
}
template<typename T, typename ItemType>
INLINE auto perform_lbound(const T &x, ItemType item) {
    return perform_lbound(x, item, has_lower_bound_mf<T>());
}

template<typename FType, typename SizeType>
struct ProjID: public std::pair<FType, SizeType> {
    static_assert(std::is_integral<SizeType>::value, "must be integral");
    static_assert(std::is_floating_point<FType>::value, "must be floating point");
    template<typename...Args>
    ProjID(Args &&...args): std::pair<FType, SizeType>(std::forward<Args>(args)...) {}
    ProjID() {}
    FType f() const {return this->first;}
    FType fabs() const {return std::abs(this->first);}
    SizeType id() const {return this->second;}
    bool operator<(FType x) const {return f() < x;}
    bool operator<=(FType x) const {return f() <= x;}
    bool operator>(FType x) const {return f() > x;}
    bool operator>=(FType x) const {return f() >= x;}
};

template<typename T>
double cossim(const T &x, const T &y) {
    auto sim = dot(x, y);
    auto xs = dot(x, x), ys = dot(x, y);
    return sim / std::sqrt(xs + ys);
}


#if 0
struct stop_param_generator {
    double v_;
    bool t_;
    stop_param_generator(double v, bool t): v_(v), t_(t) {
        if(!t) v_ = 1. - std::log2(v);
    }
    double get() const {
        if(t) {
        }
    }
};
#endif
template<typename ValueType,
         typename IdType=std::uint32_t, typename FType=std::decay_t<decltype(*std::begin(std::declval<ValueType>()))>,
         template <typename...> class map_template=std::set,
         template <typename...> class SetTemplate=ska::flat_hash_set>
class DCI;


template<typename ValueType,
         typename IdType, typename FType,
         template <typename...> class map_template,
         template <typename...> class SetTemplate>
class DCI {
    /*
     To use this: Hold values in their own container. (This is non-owning.)
     Provide a distance metric/dot function which performs dot through ADL.
     https://arxiv.org/abs/1512.00442
    */
    using ProjI = ProjID<FType, IdType>;
    using map_type = map_template<ProjI>;
    //using map_type = sorted::vector<ProjI>;
    using set_type = SetTemplate<IdType>;
    using matrix_type = blaze::DynamicMatrix<FType>;
    using value_type = ValueType;
    using bin_tree_iterator = typename map_type::const_iterator;
    matrix_type mat_;
    std::vector<map_type> map_;
    std::vector<const value_type*> val_ptrs_;
public:
    const unsigned m_, l_, d_;
private:
    size_t n_inserted_;
    double eps_;
    double gamma_ = 1.;
    int orthonormalize_:1;
    int data_dependent_:1;
public:
    size_t total() const {return m_ * l_;}
    void set_data_dependence(bool val) {
        if(val) throw std::runtime_error("Not implemented: data_dependent version");
        data_dependent_ = val;
    }
    auto &gamma() {return gamma_;}
    auto gamma() const {return gamma_;}
    auto data_dependent() const {return data_dependent_;}
    auto orthonormalize() const {return orthonormalize_;}
    auto eps() const {return eps_;}
    auto n_inserted() const {return n_inserted_;}
    auto &map() {return map_;}
    const auto &map() const {return map_;}
    auto &mat() {return mat_;}
    const auto &mat() const {return mat_;}
    template<template<typename...> class NewSetTemplate=sorted::vector>
    DCI<ValueType, IdType, FType, NewSetTemplate> cvt() const & {
        return DCI<ValueType, IdType, FType, NewSetTemplate>(*this);
    }
    template<template<typename...> class NewSetTemplate=sorted::vector>
    DCI<ValueType, IdType, FType, NewSetTemplate> cvt() const && {
        return DCI<ValueType, IdType, FType, NewSetTemplate>(*this);
    }
    template<template<typename...> class NewSetTemplate=sorted::vector>
    DCI<ValueType, IdType, FType, NewSetTemplate> cvt() && {
        return DCI<ValueType, IdType, FType, NewSetTemplate>(*this);
    }
    template<template<typename...> class NewSetTemplate=sorted::vector>
    DCI<ValueType, IdType, FType, NewSetTemplate> cvt() & {
        return DCI<ValueType, IdType, FType, NewSetTemplate>(*this);
    }
    template<template <typename...> class OldSetTemplate>
    DCI(DCI<ValueType, IdType, FType, OldSetTemplate> &&o):
        val_ptrs_(std::move(o.vps())),
        m_(o.m_), l_(o.l_), d_(o.d_), mat_(std::move(o.mat())), n_inserted_(o.n_inserted()),
        eps_(o.eps()), gamma_(o.gamma()), data_dependent_(o.data_dependent()), orthonormalize_(o.orthonormalize())
    {
        map().reserve(o.map().size());
        for(const auto &p: o.map())
            map_.emplace_back(p.begin(), p.end());
    }
    template<template <typename...> class OldSetTemplate>
    DCI(const DCI<ValueType, IdType, FType, OldSetTemplate> &o):
        val_ptrs_(o.vps()),
        m_(o.m_), l_(o.l_), d_(o.d_), mat_(o.mat()), n_inserted_(o.n_inserted()),
        eps_(o.eps()), gamma_(o.gamma()), data_dependent_(o.data_dependent()), orthonormalize_(o.orthonormalize())
    {
        map().reserve(o.map().size());
        for(const auto &p: o.map())
            map_.emplace_back(p.begin(), p.end());
    }
    DCI(size_t m, size_t l, size_t d, double eps=1e-5,
        bool orthonormalize=false, float param=1., bool dd=false):
        m_(m), l_(l), d_(d), mat_(m * l, d), map_(m * l), n_inserted_(0), eps_(eps), gamma_(param), data_dependent_(dd), orthonormalize_(orthonormalize)
    {
        blaze::randomize(mat_);
        std::fprintf(stderr, "Made mat of %zu/%zu with m, l, d as %zu, %zu, %zu\n", mat_.rows(), mat_.columns(), m, l, d);
        orthonormalize_ = false;
        if(orthonormalize_) {
            try {
                matrix_type r, q;
                blaze::qr(mat_, q, r);
                std::fprintf(stderr, "q size: %zu/%zu\n", q.rows(), q.columns());
                std::fprintf(stderr, "r size: %zu/%zu\n", r.rows(), r.columns());
                std::fprintf(stderr, "mat_ size: %zu/%zu\n", mat_.rows(), mat_.columns());
                std::cout << "q: \n\n" << q << '\n';
                std::cout << "r: \n\n" << r << '\n';
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
        auto tmp = mat_ * val;
        ProjI to_insert;
        const auto id = n_inserted_++;
        val_ptrs_.emplace_back(std::addressof(val));
        #pragma omp parallel for
        for(size_t i = 0; i < mat_.rows(); ++i) {
            map_[i].emplace(ProjI(tmp[i], id));
        }
#if 0
        std::fprintf(stderr, "ind: %u. inserted: %u. valp sz: %zu\n", ind, unsigned(n_inserted_), val_ptrs_.size());
        assert(val_ptrs_.size() == n_inserted_);
#endif
    }
    bool should_stop(size_t i, size_t candidateset_size, unsigned k) const {
        // Warning: this currenly
        const double rat = double(val_ptrs_.size()) / k;
        auto exp = 1. - std::log2(gamma_);
        const size_t ktilde = k * std::max(
            std::ceil(std::log(rat)),
            gamma_ == 1. ? rat: std::pow(rat, exp)
        );
        //const size_t ktilde = std::ceil(k * std::max(std::log(rat), std::pow(rat, 1 - std::log2(gamma_))));
        return candidateset_size >= std::min(ktilde, val_ptrs_.size());
    }
    static const ProjI *next_best(const map_type &map, std::pair<bin_tree_iterator, bin_tree_iterator> &bi, FType val) {
        if(bi.first != map.begin()) {
            if(bi.second != map.end()) {
                //std::fprintf(stderr, "dist1: %f. dist2: %f.\n", std::abs(bi.first->first - val), std::abs(bi.second->first - val));
                auto it = std::abs(bi.first->first - val) > std::abs(bi.second->first - val)
                    ? bi.first-- : bi.second++;
                if(bi.second != map.end() && bi.first != map.begin())
                {
                }
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
    using ProjIM = std::pair<ProjI, uint32_t>;
    std::vector<ProjI> prioritized_query(const ValueType &val, unsigned k, unsigned k1) const {
        if(k <= val_ptrs_.size())
            return query(val, k);
        if(k1 < 2) throw std::runtime_error("Expected k1 > 1");
        using PQT = std::priority_queue<ProjIM, std::vector<ProjIM>, std::greater<ProjIM>>;
        // Get a pair of iterators
        std::vector<std::pair<bin_tree_iterator, bin_tree_iterator>> bounds(l_ * m_);
        std::vector<PQT> pqs(l_);
        std::vector<set_type> candidates(l_);
        blaze::DynamicVector<FType> dists = mat_ * val;
        blaze::DynamicMatrix<uint32_t> countsvec(l_, size());
        std::vector<ProjI> ret;

        // Initialize queues
        OMP_PRAGMA("omp parallel for")
        for(unsigned i = 0; i < l_; ++i) {
            // Parallelize over l, avoid conflicts because there are only l 
            auto &pq = pqs[i];
            for(size_t j = 0; j < m_; ++j) {
                auto index = ind(j, i);
                map_type &pos = map_[i];
                const bin_tree_iterator it = perform_lbound(pos, dists[index]);
                auto cp = it;
                ProjI to_insert;
                const double dl = std::abs(dists[i] - it->first);
                if(likely(cp != pos.end())) {
                    const double dr = std::abs(dists[i] - ++cp->first);
                    to_insert = dl < dr ? ProjIM{{dl, it--->second}, j}: ProjIM{{dr, cp++->second}, j};
                } else to_insert = {dl, it--->second, j};
                bounds[i] = std::make_pair(it, cp);
                pq.push(to_insert);
            }
        }
        for(uint32_t k1i = 1; k1i < k1; ++k1i) {
            for(uint32_t l = 0; l < l_; ++l) {
                auto &canset = candidates[l];
                if(canset.size() >= k) continue;
                auto C = row(countsvec, l);
                auto &pq = pqs[l];
                auto top = pq.top(); pq.pop();
                auto j = top.second;
                auto index = ind(j, l);
                auto pair = next_best(map_[index], bounds[index], dists[index]);
                if(unlikely(!pair)) throw std::runtime_error("Failure in navigating tree");
                pq.push(ProjIM(ProjI(*pair), j));
                if(++C[top.first.second] == m_) {
                    canset.insert(top.first.second);
                }
            }
        }
        auto it = candidates.begin();
        set_type u = std::move(*it++);
        while(it != candidates.end()) {
            u.insert(it->begin(), it->end());
            ++it;
        }
        std::vector<ProjI> vs(k);
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
        return ret;
    }
    std::vector<ProjI> query(const ValueType &val, unsigned k) const {
        bool klt = k <= val_ptrs_.size();
        std::vector<ProjI> vs(klt ? k: unsigned(val_ptrs_.size()));
        if(!klt) {
            k = val_ptrs_.size();
            blaze::DynamicVector<FType> dists(k);
            for(size_t i = 0; i < k; ++i)
                dists[i] = norm(val - trans(row(mat_, i)));
            std::priority_queue<ProjI, std::vector<ProjI>> pq;
            for(size_t i = 0; i < dists.size(); ++i) {
                const FType tmp = dists[i];
                if(pq.size() < k) {
                    pq.push(ProjI(tmp, i));
                } else if(pq.size() == k && pq.top().fabs() > tmp) {
                    pq.pop();
                    pq.push(ProjI(tmp, i));
                }
            }
            for(int i = k; i--;pq.pop()) vs[i] = pq.top();
            return vs;
        }

        // First step: dot product the query with all reference positions
        std::vector<std::pair<bin_tree_iterator, bin_tree_iterator>> bounds(l_ * m_);
        set_type candidates;
        // Get a pair of iterators
        blaze::DynamicVector<FType> dists = mat_ * val;
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < l_ * m_; ++i) {
            const map_type &pos = map_[i];
            static_assert(std::is_same<bin_tree_iterator, typename map_type::const_iterator>::value, "must be");
            static_assert(std::is_same<typename std::remove_const<bin_tree_iterator>::type, std::decay_t<decltype(pos.begin())>>::value, "ZOMG");
            //TD<std::decay_t<decltype(pos.begin())>> td;
            const bin_tree_iterator it = perform_lbound(pos, dists[i]);
            auto cp = it;
            if(cp != pos.end()) ++cp;
            bounds[i] = std::make_pair(it, cp);
        }


        // TODO: consider sparse representation. I don't expect dense to be best.
        // Allocate counts
        blaze::DynamicMatrix<uint32_t> countsvec(l_, size());

        // Iterate through ith closest along each projection direction.
        // TODO: parallelize
        for(size_t i = 0; i < size(); ++i) {
            for(size_t l = 0; l < l_; ++l) {
                //auto &candidates = candidatesvec[l];
                auto C = row(countsvec, l);
                /* 1. Get `ith` closest to q_{jl} [the `dist` above]
                 * 2.
                 */
                for(size_t j = 0; j < m_; ++j) {
                    auto index = ind(j, l);
                    auto pair = next_best(map_[index], bounds[index], dists[index]);
                    if(!pair) throw std::runtime_error("Failure in navigating tree");
                    if(++C[pair->second] == m_) {
                        candidates.insert(pair->second);
                    }
                }
            }
            if(i == size() - 1) std::fprintf(stderr, "ran through all iterations\n");
            if(should_stop(i, candidates.size(), k)) break;
        }
        auto &u = candidates;
        std::fprintf(stderr, "Candidates size: %zu\n", u.size());
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
    struct iterator: public std::vector<const ValueType *>::iterator {
        using super = typename std::vector<const ValueType *>::iterator;
        template<typename...Args>
        iterator(Args&&...args): super(std::forward<Args>(args)...) {}
        ValueType *operator->() {
            return super::operator->();
        }
        const ValueType *operator->() const {
            return super::operator->();
        }
    };
    struct const_iterator: public std::vector<const ValueType *>::const_iterator {
        using super = typename std::vector<const ValueType *>::const_iterator;
        template<typename...Args>
        const_iterator(Args&&...args): super(std::forward<Args>(args)...) {}
        const ValueType *operator->() const {
            return super::operator->();
        }
    };
    auto begin() {return iterator(val_ptrs_.begin());}
    auto end() {return iterator(val_ptrs_.end());}
    auto begin() const {return const_iterator(val_ptrs_.begin());}
    auto end()   const {return const_iterator(val_ptrs_.end());}
    auto &&vps() && {return std::move(val_ptrs_);}
    const auto & vps() const && {return val_ptrs_;}
    auto &vps()  & {return val_ptrs_;}
    const auto & vps() const & {return val_ptrs_;}
};

} // dci

} // namespace frp

#endif
