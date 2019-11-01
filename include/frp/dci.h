#ifndef FRP_DCI_H__
#define FRP_DCI_H__
#ifndef FHT_HEADER_ONLY
#  define FHT_HEADER_ONLY 1
#endif
#include <cstdint>
#include <cstdlib>
#include "./ifc.h"
#include "./util.h"
#include "lsh.h"
#include "sdq.h"
#include <deque>
#include "blaze/Math.h"
#include <map>
#include <cmath>
#include <set>
#include <queue>
#include <vector>
#include "flat_hash_map/flat_hash_map.hpp"

namespace lb {
template<typename T>
struct has_lower_bound_mf: std::false_type {};
template<typename T>
struct has_lower_bound_mf<std::set<T>>: std::true_type {};
}
namespace frp {

namespace dci {

template<typename FloatType, bool SO, typename T>
auto dot(const blaze::DynamicVector<FloatType, SO> &r, const T &x) {
    return blaze::dot(r, x);
}
using namespace lb;


template<typename T, typename ItemType>
INLINE auto perform_lbound(const T &x, ItemType item, std::false_type) {
    return std::lower_bound(std::begin(x), std::end(x), item);
}

template<typename T, typename ItemType>
INLINE auto perform_lbound(const T &x, ItemType item, std::true_type) {
    return x.lower_bound(item);
}
template<typename T, typename ItemType>
INLINE auto perform_lbound(const T &x, ItemType item) {
    return perform_lbound(x, item, has_lower_bound_mf<T>());
}

static_assert(has_lower_bound_mf<std::set<int>>::value, "std::set must have lb mf");

template<typename FType, typename SizeType>
struct ProjID: public std::pair<FType, SizeType> {
    static_assert(std::is_integral<SizeType>::value, "must be integral");
    static_assert(std::is_floating_point<FType>::value, "must be floating point");
    template<typename...Args>
    ProjID(Args &&...args): std::pair<FType, SizeType>(std::forward<Args>(args)...) {}
    ProjID() {}
    //ProjID(FType x): std::pair<FType, SizeType>(x, 0) {}
    FType f() const {return this->first;}
    FType fabs() const {return std::abs(this->first);}
    SizeType id() const {return this->second;}
    bool operator<(FType x) const {return f() < x;}
    bool operator<=(FType x) const {return f() <= x;}
    bool operator>(FType x) const {return f() > x;}
    bool operator>=(FType x) const {return f() >= x;}
};

template<typename FT, typename ST>
INLINE bool operator<(FT x, ProjID<FT, ST> y) {
    return x < y.f();
}
template<typename FT, typename ST>
INLINE bool operator<=(FT x, ProjID<FT, ST> y) {
    return x <= y.f();
}
template<typename FT, typename ST>
INLINE bool operator>=(FT x, ProjID<FT, ST> y) {
    return x >= y.f();
}
template<typename FT, typename ST>
INLINE bool operator>(FT x, ProjID<FT, ST> y) {
    return x > y.f();
}

template<typename IType, typename Set, typename VT>
std::pair<IType, IType> get_iterator_pair(const Set &set, IType it, VT pv) {
    bool ibeg = it == set.begin(), iend = it == set.end();
    auto lit = it, rit = it;
    if(!ibeg) --lit;
    else lit = set.end();
    if(!iend) ++rit;
    std::pair<IType, IType> ret;
    if(rit == set.end()) {
        ret = {lit, it};
    }
    else if(lit == set.end()) {
        ret = {it, rit};
    }
    else if(std::abs(lit->first - pv) > std::abs(rit->first - pv)) {
        ret = {it, rit};
    } else ret = {lit, it};
    return ret;
}

template<typename T>
double cossim(const T &x, const T &y) {
    auto sim = dot(x, y);
    auto xs = dot(x, x), ys = dot(x, y);
    return sim / std::sqrt(xs + ys);
}

template<typename VT, typename Allocator=std::allocator<VT>>
struct vless_set: public std::set<VT, std::less<>, Allocator> {
    template<typename...Args>
    vless_set(Args &&...args): std::set<VT, std::less<>, Allocator>(std::forward<Args>(args)...) {};
};

template<typename ValueType,
         typename IdType=std::uint32_t, typename FType=std::decay_t<decltype(*std::begin(std::declval<ValueType>()))>,
         template <typename...> class MapTemplate=vless_set,
         template <typename...> class SetTemplate=ska::flat_hash_set,
         bool SO=blaze::rowMajor,
         typename Projector=MatrixLSHasher<FType, SO>,
         typename CMatType=std::uint16_t>
class DCI;


template<typename ValueType,
         typename IdType, typename FType,
         template <typename...> class MapTemplate,
         template <typename...> class SetTemplate,
         bool SO,
         typename Projector,
         typename CMatType>
class DCI {
    using this_type = DCI<ValueType, IdType, FType, MapTemplate, SetTemplate, SO, Projector, CMatType>;
    using const_this_type = const DCI<ValueType, IdType, FType, MapTemplate, SetTemplate, SO, Projector, CMatType>;
    /*
     To use this: Hold values in their own container. (This is non-owning.)
     Provide a distance metric/dot function which performs dot through ADL.
     https://arxiv.org/abs/1512.00442
    */
    using ProjI = ProjID<FType, IdType>;
    using map_type = MapTemplate<ProjI>;
    //using map_type = sorted::vector<ProjI>;
    using set_type = SetTemplate<IdType>;
    using matrix_type = blaze::DynamicMatrix<FType, SO>;
    using value_type = ValueType;
    using bin_tree_iterator = typename map_type::const_iterator;
    using float_type = std::decay_t<decltype(*std::begin(std::declval<ValueType>()))>;
    std::vector<map_type> map_;
    std::vector<const value_type*> val_ptrs_;
public:
    const unsigned m_, l_, d_;
private:
    Projector proj_;
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
    auto &proj() {return proj_;}
    const auto &proj() const {return proj_;}
    template<template<typename...> class NewSetTemplate=sorted::vector>
    auto cvt() const & {
        return DCI<ValueType, IdType, FType, MapTemplate, NewSetTemplate, SO, Projector, CMatType>(*this);
    }
    template<template<typename...> class NewSetTemplate=sorted::vector>
    auto cvt() const && {
        return DCI<ValueType, IdType, FType, MapTemplate, NewSetTemplate, SO, Projector, CMatType>(std::move(const_cast<this_type &&>(*this)));
    }
    template<template<typename...> class NewSetTemplate=sorted::vector>
    auto cvt() && {
        return DCI<ValueType, IdType, FType, MapTemplate, NewSetTemplate, SO, Projector, CMatType>(std::move(*this));
    }
    template<template<typename...> class NewSetTemplate=sorted::vector>
    auto cvt() & {
        return DCI<ValueType, IdType, FType, MapTemplate, NewSetTemplate, SO, Projector, CMatType>(*this);
    }
    template<template <typename...> class OldSetTemplate>
    DCI(DCI<ValueType, IdType, FType, MapTemplate, OldSetTemplate, SO, Projector, CMatType> &&o):
        val_ptrs_(std::move(o.vps())),
        m_(o.m_), l_(o.l_), d_(o.d_), proj_(std::move(o.proj())), n_inserted_(o.n_inserted()),
        eps_(o.eps()), gamma_(o.gamma()),
        orthonormalize_(o.orthonormalize()),
        data_dependent_(o.data_dependent())
    {
        map().reserve(o.map().size());
        for(const auto &p: o.map())
            map_.emplace_back(p.begin(), p.end());
    }
    template<template <typename...> class OldSetTemplate>
    DCI(const DCI<ValueType, IdType, FType, MapTemplate, OldSetTemplate, SO, Projector, CMatType> &o):
        val_ptrs_(o.vps()),
        m_(o.m_), l_(o.l_), d_(o.d_), proj_(o.proj()), n_inserted_(o.n_inserted()),
        eps_(o.eps()), gamma_(o.gamma()),
        orthonormalize_(o.orthonormalize()),
        data_dependent_(o.data_dependent())
    {
        map().reserve(o.map().size());
        for(const auto &p: o.map())
            map_.emplace_back(p.begin(), p.end());
    }
    DCI(size_t m, size_t l, size_t d, double eps=1e-5,
        bool orthonormalize=true, float param=1., bool dd=false, uint64_t seed=1337):
        map_(m * l),
        m_(m), l_(l), d_(d),
        proj_(m * l, d, orthonormalize, seed),
        n_inserted_(0), eps_(eps), gamma_(param),
        orthonormalize_(orthonormalize),
        data_dependent_(dd)
    {
    }
#if 0
    template<bool OSO>
    void insert(const blaze::DynamicMatrix<float_type, OSO> &o) {
        val_ptrs_.reserve(val_ptrs_.size() + o.rows());
        for(size_t i = 0; i < o.rows();++i) {
            throw std::runtime_error("This doesn't work as-is. We'd need to update the indexing step to take spans");
            auto r = row(o, i);
            val_ptrs_.emplace_back(reinterpret_cast<ValueType *>(&r));
        }
        auto tmp = proj_.project(o);
        assert(tmp.rows() == mat_.rows()); // This is true because maths
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < mat_.rows(); ++i) {
            auto &map = map_[i];
            auto r = row(tmp, i);
            for(size_t offset = 0; offset < o.rows(); ++offset) {
                map.emplace(ProjI(r[i], offset + n_inserted_));
            }
        }
        n_inserted_ += o.rows();
    }
#endif
    template<typename I>
    void insert(I i1, I i2) {
        while(i1 != i2)
            add_item(*i1++);
    }
    void add_item(const ValueType &val) {
        auto tmp = proj_.project(val);
        ProjI to_insert;
        const auto id = n_inserted_++;
        val_ptrs_.emplace_back(std::addressof(val));
        #pragma omp parallel for
        for(size_t i = 0; i < m_ * l_; ++i) {
            map_[i].emplace(ProjI(tmp[i], id));
        }
    }
    bool should_stop(size_t candidateset_size, unsigned k) const {
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
    static const ProjI *next_best(const map_type &s,
            std::pair<bin_tree_iterator, bin_tree_iterator> &its, FType v) {
        if(its.first != s.end()) {
            const bool beg = its.first == s.begin();
            if(its.second != s.end()) {
                bool usefirst = std::abs(its.first->first - v) < std::abs(its.second->first - v);
                bin_tree_iterator it;
                if(usefirst) {
                    it = its.first;
                    if(!beg) --its.first;
                    else its.first = s.end();
                } else {
                    it = its.second++;
                }
                return &*it;
            }
            auto it = its.first;
            if(beg) {
                its.first = s.end();
            } else {
                --its.first;
            }
            return &*it;
        } else if(its.second != s.end()) {
            auto it = its.second++;
            return &*it;
        }
        return nullptr;
    }
#if 0
    static const ProjI *next_best(const map_type &map, std::pair<bin_tree_iterator, bin_tree_iterator> &bi, FType val) {
        std::fprintf(stderr, "TODO: write a unit test to ensure that we visit this in order\n");
        if(bi.first != map.end()) {
            const bool beg = bi.first == map.begin();
            if(bi.second != map.end()) {
                //std::fprintf(stderr, "dist1: %f. dist2: %f.\n", std::abs(bi.first->first - val), std::abs(bi.second->first - val));
                auto it = std::abs(bi.first->first - val) > std::abs(bi.second->first - val)
                    ? bi.first : bi.second++;
                if(beg)
                    bi.first == map.end();
                else --bi.first;
                return &*it;
            }
            auto it = bi.first;
            if(beg)
                bi.first == map.end();
            else
                --bi.first;
            return &*it;
        } else if(bi.second != map.end()) {
            auto it = bi.second++;
            return &*it;
        }
        return nullptr;
    }
#endif
    std::vector<ProjI> prioritized_query(const ValueType &val, unsigned k, unsigned k1) const {
        using ProjIM = std::pair<ProjI, IdType>; // To track 'm', as well, as that's necessary for prioritized query.
        if(k <= val_ptrs_.size())
            return query(val, k);
        if(k1 < 2) throw std::runtime_error("Expected k1 > 1");
        std::fprintf(stderr, "k: %u. k1: %u\n", k, k1);
        using PQT = std::priority_queue<ProjIM, std::vector<ProjIM>, std::greater<ProjIM>>;
        // Get a pair of iterators
        std::vector<std::pair<bin_tree_iterator, bin_tree_iterator>> bounds(l_ * m_);
        std::vector<PQT> pqs(l_);
        std::vector<set_type> candidates(l_);
        auto projections = proj_.project(val);
        blaze::DynamicMatrix<CMatType> countsvec(l_, size(), 0);
        std::vector<ProjI> ret;

        // Initialize queues
        OMP_PRAGMA("omp parallel for")
        for(unsigned i = 0; i < l_; ++i) {
            // Parallelize over l, avoid conflicts because there are only l
            auto &pq = pqs[i];
            for(size_t j = 0; j < m_; ++j) {
                auto index = ind(j, i);
                const map_type &pos = map_[i];
                const auto pv = projections[index];
                bin_tree_iterator it = perform_lbound(pos, projections[index]);
                bounds[i] = get_iterator_pair(pos, it, pv);
                auto getv = [pv,e=pos.end()](auto x) {return x != e ? std::abs(x->first - pv): std::numeric_limits<double>::max();};
                auto lv = getv(bounds[i].first), rv = getv(bounds[i].second);
                if(lv < rv) {
                    pq.push(ProjIM(ProjI(lv, bounds[i].first->second), j));
                } else {
                    pq.push(ProjIM(ProjI(rv, bounds[i].second->second), j));
                }
            }
        }
        for(uint32_t k1i = 1; k1i < k1; ++k1i) {
            for(uint32_t l = 0; l < l_; ++l) {
                auto &canset = candidates[l];
                if(canset.size() >= k) continue;
                auto C = row(countsvec, l);
                auto &pq = pqs[l];

                // Top of the pops
                auto top = pq.top();
                pq.pop();
                auto j = top.second;
                auto index = ind(j, l);
                // Get next best
                auto pair = next_best(map_[index], bounds[index], projections[index]);
                if(unlikely(!pair)) throw std::runtime_error("Failure in navigating tree");
                pq.push(ProjIM(ProjI(*pair), j));
                if(++C[top.first.second] == m_) {
                    canset.insert(top.first.second);
                }
            }
        }
        auto it = candidates.begin();
        set_type u = std::move(*it);
        while(++it != candidates.end()) {
            u.insert(it->begin(), it->end());
        }
        std::vector<ProjI> vs(k);
        std::priority_queue<ProjI> pq;
        for(auto it = u.begin(); it != u.end(); ++it) {
            auto p = val_ptrs_[*it];
            FType tmp = blaze::norm(*p - val);
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
            OMP_PRAGMA("omp parallel for")
            for(size_t i = 0; i < k; ++i)
                dists[i] = norm(val - this->operator[](i));
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
            for(int i = k; i--; pq.pop()) vs[i] = pq.top();
            return vs;
        }

        // First step: dot product the query with all reference positions
        std::vector<std::pair<bin_tree_iterator, bin_tree_iterator>> bounds(l_ * m_);
        set_type candidates;
        // Get a pair of iterators
        blaze::DynamicVector<FType> projections = proj_.project(val);
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < l_ * m_; ++i) {
            const map_type &pos = map_[i];
            static_assert(std::is_same<bin_tree_iterator, typename map_type::const_iterator>::value, "must be");
            static_assert(std::is_same<typename std::remove_const<bin_tree_iterator>::type, std::decay_t<decltype(pos.begin())>>::value, "ZOMG");
            //TD<std::decay_t<decltype(pos.begin())>> td;
            const auto pv = projections[i];
            bounds[i] = get_iterator_pair(pos, perform_lbound(pos, pv), pv);
        }


        // TODO: consider sparse representation. I don't expect dense to be best.
        // Allocate counts
        blaze::DynamicMatrix<CMatType> countsvec(l_, size(), 0);

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
                    auto pair = next_best(map_[index], bounds[index], projections[index]);
                    if(!pair) throw std::runtime_error("Failure in navigating tree");
                    if(++C[pair->second] == m_) {
                        candidates.insert(pair->second);
                    }
                    assert(C[pair->second] ||
                            !std::fprintf(stderr, "Note: we may have overflowed CMatType limit, as this should not be zero.")
                    ); // Ensure that we haven't overflowed
                }
            }
            if(should_stop(candidates.size(), k)) break;
        }
        auto &u = candidates;
#if !NDEBUG
        std::fprintf(stderr, "Candidates size: %zu\n", u.size());
#endif
        std::priority_queue<ProjI> pq;
        for(auto it = u.begin(); it != u.end(); ++it) {
            FType tmp = blaze::norm(this->operator[](*it) - val);
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
    const value_type &operator[](size_t index) const {return *val_ptrs_[index];}
    value_type &operator[](size_t index) {return *val_ptrs_[index];}
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
