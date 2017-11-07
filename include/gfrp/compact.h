#ifndef _GFRP_CRAD_H__
#define _GFRP_CRAD_H__
#include "gfrp/util.h"
#include "gfrp/linalg.h"
#include "fastrange/fastrange.h"
#include <ctime>

namespace gfrp {

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

// TODO!!!
/*
   Add random-access AES-CTR-based Rademacher array.
   Add random-access std::*distribution array.

   Need help with ideas:
      1. Pruning or selecting more important random features:::Talk to Arora.
      2.

 */

struct free_delete {
    void operator()(void *ptr) const {free(ptr);}
};

template<typename FloatType=FLOAT_TYPE, typename T=uint64_t, typename RNG=aes::AesCtr<T>, typename=enable_if_t<is_arithmetic<FloatType>::value>>
class CompactRademacher {

    size_t n_, m_;
    unique_ptr<T, free_delete> data_;

    static constexpr FloatType values_[2] {1, -1};
    static constexpr size_t NBITS = sizeof(T) * CHAR_BIT;
    static constexpr size_t SHIFT = log2_64(NBITS);
    static constexpr size_t BITMASK = NBITS - 1;

    using value_type = FloatType;
    using container_type = T;
    using size_type = size_t;
public:
    // Constructors
    CompactRademacher(size_t n=0, uint64_t seed=std::time(nullptr)): n_{n >> SHIFT}, m_{n_}, data_(static_cast<T *>(malloc(sizeof(T) * n_))) {
        if(n & (BITMASK))
            throw runtime_error(ks::sprintf("Warning: n is not evenly divisible by BITMASK size. (n: %zu). (bitmask: %zu)\n", n, BITMASK).data());
        randomize(seed);
    }
    CompactRademacher(CompactRademacher &&other) = default;
    CompactRademacher(const CompactRademacher &other): n_(other.n_), m_(other.m_), data_(static_cast<T*>(malloc(sizeof(T) * m_))) {
        if(data_ == nullptr) throw bad_alloc();
        memcpy(data_.get(), other.data_, sizeof(T) * n_);
    }
    template<typename AsType>
    class CompactAs {
        static constexpr AsType values[2] {static_cast<AsType>(1), static_cast<AsType>(-1)};
        const CompactRademacher &ref_;
    public:
        CompactAs(const CompactRademacher &ref): ref_(ref) {}
        AsType operator[](size_t index) const {return values[ref_.bool_idx(index)];}
    };
    template<typename AsType>
    CompactAs<AsType> as_type() {
        return CompactAs<AsType>(*this);
    }
    // For setting to random values
    auto *data() {return data_;}
    const auto *data() const {return data_;}
    // For use
    auto size() const {return n_ << SHIFT;}
    auto capacity() const {return m_ << SHIFT;}
    auto nwords() const {return n_;}
    auto nbytes() const {return size();}
    template<typename OWordType, typename OFloatType>
    bool operator==(const CompactRademacher<OWordType, OFloatType> &other) const {
        if(size() != other.size()) return false;
        auto odata = other.data();
        for(size_t i(0);i < n_; ++i)
            if(data_[i] != odata[i])
                return false;
        return true;
    }

    void randomize(uint64_t seed) {
        random_fill(reinterpret_cast<uint64_t *>(data_.get()), n_ * sizeof(uint64_t) / sizeof(T), seed);
    }
    void zero() {memset(data_, 0, sizeof(T) * (n_ >> SHIFT));}
    void reserve(size_t newsize) {
        if(newsize & (newsize - 1)) throw runtime_error("newsize should be a power of two");
        if(newsize > m_) {
            auto tmp(static_cast<T*>(realloc(data_, sizeof(T) * (newsize >> SHIFT))));
            if(tmp == nullptr) throw bad_alloc();
            data_ = tmp;
        }
    }
    int bool_idx(size_type idx) const {return !(data_.get()[(idx >> SHIFT)] & (static_cast<T>(1) << (idx & BITMASK)));}

    FloatType operator[](size_type idx) const {return values_[bool_idx(idx)];}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        static_assert(is_same<decay_t<decltype(in[0])>, FloatType>::value, "Input vector should be the same type as this structure.");
        static_assert(is_same<decay_t<decltype(out[0])>, FloatType>::value, "Output vector should be the same type as this structure.");
        throw runtime_error("Not Implemented.");
    }
};

template<typename SizeType=size_t, typename RNG=aes::AesCtr<uint64_t>>
class OnlineShuffler {
    //Provides reproducible shuffling by re-generating a random sequence for shuffling an array.
    //This
    using ResultType = typename RNG::result_type;
    const uint64_t seed_;
    RNG             rng_;
public:
    explicit OnlineShuffler(ResultType seed): seed_{seed}, rng_(seed) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        fprintf(stderr, "[W:%s] OnlineShuffler can only shuffle from arrays of different sizes by sampling.\n");
        const auto isz(in.size());
        if(isz == out.size()) {
            out = in;
            apply<OutVector>(out);
        } else if(isz > out.size()) {
            unordered_set<uint64_t> indices;
            indices.reserve(out.size());
            while(indices.size() < out.size()) indices.insert(fastrange64(rng_(), isz));
            auto it(out.begin());
            for(const auto index: indices) *it++ = in[index]; // Could consider a sorted map for quicker iteration/cache coherence.
        } else {
            for(auto it(out.begin()), eit(out.end()); it != eit;)
                *it++ = in[fastrange64(rng_(), isz)];
        }
        //The naive approach is double memory.
    }
    template<typename Vector>
    void apply(Vector &vec) const {
        rng_.seed(seed_);
        shuffle(begin(vec), end(vec), rng_);
    }
};

struct UnchangedRNGDistribution {
    template<typename RNG>
    auto operator()(RNG &rng) const {return rng();}
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
    struct emit_u64 {
        RNG rng_;
        using result_type = typename RNG::result_type;
        static constexpr result_type max() {return RNG::max();}
        static constexpr result_type min() {return RNG::min();}
        void set(result_type index) {
            rng_.fast_forward(index);
        }
        result_type operator()() {return rng_();}
    } emitter_;


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



    template<typename=enable_if_t<aes::is_aes<RNG>::value>>
    ResultType operator[](size_t index) {
#if !NDEBUG
        auto tmp(rng_[index]);
        emitter_.set(tmp);
        auto ret(dist_(emitter_));
        return ret;
#else
        emitter_.set(rng_[index]);
        return dist_(emitter_);
#endif
    }

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


} // namespace gfrp


#endif // #ifndef _GFRP_CRAD_H__
