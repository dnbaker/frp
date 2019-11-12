#ifndef _GFRP_CRAD_H__
#define _GFRP_CRAD_H__
#include "frp/util.h"
#include "frp/linalg.h"
#include "frp/dist.h"
#include "fastrange/fastrange.h"
#include <ctime>

namespace frp {

/*
// From https://arxiv.org/pdf/1702.08159.pdf

FHT!!!
https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Pseudocode
http://fourier.eng.hmc.edu/e161/lectures/wht/node4.html
Sliding windows http://www.ee.cuhk.edu.hk/~wlouyang/FWHT.htm

Presentation http://c.csie.org/~itct/slide/DCT_larry.pdf

Using renoramlization makes it orthogonal, which is GOOD.
some authors further multiply the X0 term by 1/√2 and multiply the resulting matrix by an overall scale factor of 2 N {\displaystyle {\sqrt {\tfrac {2}{N}}}} {\displaystyle {\sqrt {\tfrac {2}{N}}}} (see below for the corresponding change in DCT-III). This makes the DCT-II matrix orthogonal, but breaks the direct correspondence with a real-even DFT of half-shifted input.


Fast DCT https://unix4lyfe.org/dct-1d/
http://ieeexplore.ieee.org/document/558495/



F2F
proceeds with vectorized sums and subtractions iteratively for the first n/2^k
positions (where n is the length of the input vector and k is the iteration starting from 1)
computing the intermediate operations of the Cooley-Tukey algorithm till a small Hadamard
routine that fits in cache.  Then the algorithm continues in the same way but starting from
the  smallest  length  and  doubling  on  each  iteration  the  input  dimension  until  the  whole
FWHT is done in-place.
*/

#if __GNUC__ || __clang__
constexpr INLINE unsigned clz(unsigned long long x) {
    return __builtin_clzll(x);
}
constexpr INLINE unsigned clz(unsigned long x) {
    return __builtin_clzl(x);
}
constexpr INLINE unsigned clz(unsigned x) {
    return __builtin_clz(x);
}
#else

#define clztbl(x, arg) do {\
    switch(arg) {\
        case 0:                         x += 4; break;\
        case 1:                         x += 3; break;\
        case 2: case 3:                 x += 2; break;\
        case 4: case 5: case 6: case 7: x += 1; break;\
    }} while(0)

constexpr INLINE int clz_manual( uint32_t x )
{
  int n(0);
  if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
  if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (32 - 4));
  return n;
}

// Overload
constexpr INLINE int clz_manual( uint64_t x )
{
  int n(0);
  if ((x & 0xFFFFFFFF00000000ull) == 0) {n  = 32; x <<= 32;}
  if ((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (64 - 4));
  return n;
}
#define clz(x) clz_manual(x)
#endif

template<typename T>
static constexpr INLINE unsigned ilog2(T x) noexcept {
    return sizeof(T) * CHAR_BIT - clz(x)  - 1;
}
static constexpr unsigned log2_64(uint64_t x) {return ilog2(x);}

class PRNRademacher {
    size_t      n_;
    uint64_t seed_;
public:
    using size_type = std::size_t;
    PRNRademacher(size_t n=0, uint64_t seed=0): n_(n), seed_(seed) {}
    auto size() const {return n_;}
    void resize(size_t newsize) {n_ = newsize;}
    void seed(uint64_t seed) {seed_ = seed;}

    template<typename Container>
    void apply(Container &c) const {
        wy::WyHash<uint64_t> gen(seed_);
        uint64_t val(gen());
        using ArithType = std::decay_t<decltype(c[0])>;
        for(size_t i(0), e(c.size()); i < e; ++i) {
            if(unlikely((i & ((CHAR_BIT * sizeof(uint64_t)) - 1)) == 0))
                val = gen();
            c[i] *= val & 1 ? -1.: 1.;
            val >>= 1;
        }
    }

    template<typename ArithType>
    void apply(ArithType *c, size_t nitems=0) {
        wy::WyHash<uint64_t> gen(seed_);
        uint64_t val;
        if(nitems == 0) nitems = n_;
        for(size_t i(0); i < nitems; ++i) {
            if(unlikely((i & ((CHAR_BIT * sizeof(uint64_t)) - 1)) == 0))
                val = gen();
            c[i] *= val & 1 ? -1.: 1.;
            val >>= 1;
        }
    }
};

template<typename FT>
class CachedRademacher {
protected:
    size_t n_;
    uint64_t seed_;
    blaze::DynamicVector<FT> vec_;
public:
    using size_type = std::size_t;
    CachedRademacher(size_t n, uint64_t seed=0): n_(n), seed_(seed), vec_(n) {
        this->seed(seed);
    }
    void resize(size_t newsz) {
        if(newsz < n_) {
            vec_.resize(n_);
            n_ = newsz;
            return;
        }
        if(newsz > n_) {
            vec_.resize(newsz);
            n_ = newsz;
            seed(seed_);
        }
    }
    void seed(uint64_t seed) {
        wy::WyHash<uint64_t> gen(seed);
        unsigned t = 64;
        auto v = gen();
        for(size_t i = 0, e = n_; i < e; ++i) {
            vec_[i] = v & 1 ? -1.: 1.;
            if(t-- == 0) {
                v = gen();
                t = 64;
            }
        }
    }
    auto size() const {return n_;}
    template<typename Container>
    void apply(Container &c) const {
        //if constexpr(blaze::TransposeFlag_v<Container> == blaze::TransposeFlag_v<decltype(vec)>)
            c *= vec_;
        //else
        //    c *= trans(vec_);
    }
};

template<typename T=uint64_t, typename RNG=aes::AesCtr<T>>
class CompactRademacherTemplate {
    T              seed_;
    std::vector<T> data_;
    using FloatType = FLOAT_TYPE;

    static constexpr size_t NBITS = sizeof(T) * CHAR_BIT;
    static constexpr size_t SHIFT = log2_64(NBITS);
    static constexpr size_t BITMASK = NBITS - 1;

public:
    using value_type = FloatType;
    using container_type = T;
    using size_type = size_t;
    // Constructors
    CompactRademacherTemplate(size_t n=0, uint64_t seed=std::time(nullptr)): seed_(seed), data_(n >> SHIFT) {
        if(n & (BITMASK))
            std::fprintf(stderr, "Warning: n is not evenly divisible by BITMASK size. (n: %zu). (bitmask: %zu)\n", n, BITMASK);
        randomize(seed_);
    }
    CompactRademacherTemplate(CompactRademacherTemplate &&other) = default;
    CompactRademacherTemplate(const CompactRademacherTemplate &other) = default;
    CompactRademacherTemplate &operator=(const CompactRademacherTemplate &o) = default;
    CompactRademacherTemplate &operator=(CompactRademacherTemplate &&o) = default;
    template<typename AsType>
    class CompactAs {
        static constexpr AsType values[2] {static_cast<AsType>(1), static_cast<AsType>(-1)};
        const CompactRademacherTemplate &ref_;
    public:
        CompactAs(const CompactRademacherTemplate &ref): ref_(ref) {}
        AsType operator[](size_t index) const {return values[ref_.bool_idx(index)];}
    };
    template<typename AsType>
    CompactAs<AsType> as_type() const {
        return CompactAs<AsType>(*this);
    }
    void seed(T seed) {seed_ = seed;}
    void resize(T new_size) {
        if(new_size != size()) {
            data_.resize(std::max(static_cast<T>(1), new_size >> SHIFT));
            randomize(seed_);
        }
    }
    // For setting to random values
    auto *data() {return data_;}
    const auto *data() const {return data_;}
    // For use
    auto size() const {return data_.size() << SHIFT;}
    auto capacity() const {return data_.capacity() << SHIFT;}
    auto nwords() const {return data_.size();}
    auto nbytes() const {return size();}
    bool operator==(const CompactRademacherTemplate &other) const {
        if(size() != other.size()) return false;
        auto odata = other.data();
        for(size_t i(0);i < data_.size(); ++i)
            if(data_[i] != odata[i])
                return false;
        return true;
    }

    void randomize(uint64_t seed) {
        random_fill(reinterpret_cast<uint64_t *>(data_.data()), data_.size() * sizeof(T) / sizeof(uint64_t), seed);
    }
    void zero() {memset(data_, 0, sizeof(T) * data_.size());}
    void reserve(size_t newsize) {
        data_.reserve(newsize >> SHIFT);
    }
    INLINE int bool_idx(size_type idx) const {return !(data_[(idx >> SHIFT)] & (static_cast<T>(1) << (idx & BITMASK)));}

    FloatType operator[](size_type idx) const {return bool_idx(idx) ? FloatType(-1.): FloatType(1.);}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        static_assert(is_same<decay_t<decltype(in[0])>, FloatType>::value, "Input vector should be the same type as this structure.");
        static_assert(is_same<decay_t<decltype(out[0])>, FloatType>::value, "Output vector should be the same type as this structure.");
        out = in;
        apply(out);
    }
    template<typename FloatType2>
    void apply(FloatType2 *vec) const {
        auto tmp(as_type<FloatType2>());
        for(T i = 0; i < size(); ++i) vec[i] *= tmp[i];
    }
    template<typename VectorType>
    void apply(VectorType &vec) const {
        //std::fprintf(stderr, "Applying %s vector of size %zu.\n", __PRETTY_FUNCTION__, vec.size());
        auto tmp(as_type<std::decay_t<decltype(vec[0])>>());
        if(vec.size() != size()) {
            if(vec.size() > size())
                throw std::runtime_error("Vector is too big for he gotdam feet");
            std::fprintf(stderr, "Warning: CompactRademacherTemplate is too big. Only affecting elements in my size (%zu) vs vector size (%zu). Any F*Ts might not be so kind.\n", size_t(size()), size_t(vec.size()));
        }
        for(T i = 0, e(std::min(vec.size(), size())); i < e; ++i) {
            vec[i] *= tmp[i];
        }
    }
};

using CompactRademacher = CompactRademacherTemplate<uint64_t>;

struct UnchangedRNGDistribution {
    template<typename RNG>
    auto operator()(RNG &rng) const {return rng();}
    void reset() {}
};

struct Int2GaussianDistribution {
    template<typename RNG>
    auto operator()(RNG &rng) const {return random_gaussian_from_seed(rng());}
    void reset() {}
};

template<typename RNG=aes::AesCtr<uint64_t>, typename Distribution=UnchangedRNGDistribution>
class PRNVector {
    // Vector of random values generated
    const uint64_t    seed_;
    uint64_t          used_;
    uint64_t          size_;
    RNG                rng_;
    Distribution      dist_;
public:
    using ResultType = decay_t<decltype(dist_(rng_))>;
private:
    ResultType      val_;

public:

    class PRNIterator {

        PRNVector<RNG, Distribution> *const ref_;
    public:
        auto operator*() const {return ref_->val_;}
        auto &operator ++() {
            inc();
            return *this;
        }
        void inc() {
            ref_->gen();
            ++ref_->used_;
        }
        void gen() {ref_->gen();}
        bool operator !=([[maybe_unused]] const PRNIterator &other) const {
            return ref_->used_ < ref_->size_; // Doesn't even access the other iterator. Only used for `while(it < end)`.
        }
        PRNIterator(PRNVector<RNG, Distribution> *prn_vec): ref_(prn_vec) {}
    };

    template<typename... DistArgs>
    PRNVector(uint64_t size, uint64_t seed=0, DistArgs &&... args):
        seed_{seed}, used_{0}, size_{size}, rng_(seed_), dist_(forward<DistArgs>(args)...), val_(gen()) {}

    auto begin() {
        reset();
        return PRNIterator(this);
    }
    ResultType gen() {return val_ = dist_(rng_);}
    void reset() {
        rng_.seed(seed_);
        dist_.reset();
        used_ = 0;
        gen();
    }
    auto end() {
        return PRNIterator(static_cast<decltype(this)>(nullptr));
    }
    auto end() const {
        return PRNIterator(static_cast<decltype(this)>(nullptr));
    }
    auto size() const {return size_;}
    void resize(size_t newsize) {size_ = newsize;}
};

} // namespace frp


#endif // #ifndef _GFRP_CRAD_H__
