#ifndef _GFRP_CRAD_H__
#define _GFRP_CRAD_H__
#include "gfrp/util.h"
#include "gfrp/linalg.h"
#include "gfrp/dist.h"
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

template<typename T=uint64_t, typename RNG=aes::AesCtr<T>>
class CompactRademacherTemplate {
    T              seed_;
    std::vector<T> data_;
    using FloatType = FLOAT_TYPE;

    static constexpr FloatType values_[2] {1, -1};
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
    template<typename AsType>
    class CompactAs {
        static constexpr AsType values[2] {static_cast<AsType>(1), static_cast<AsType>(-1)};
        const CompactRademacherTemplate &ref_;
    public:
        CompactAs(const CompactRademacherTemplate &ref): ref_(ref) {}
        AsType operator[](size_t index) const {return values[ref_.bool_idx(index)];}
    };
    template<typename AsType>
    CompactAs<AsType> as_type() {
        return CompactAs<AsType>(*this);
    }
    void seed(T seed) {seed_ = seed;}
    void resize(T new_size) {
        if(new_size != size()) {
            data_.resize(new_size >> SHIFT);
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
    int bool_idx(size_type idx) const {return !(data_[(idx >> SHIFT)] & (static_cast<T>(1) << (idx & BITMASK)));}

    FloatType operator[](size_type idx) const {return values_[bool_idx(idx)];}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        static_assert(is_same<decay_t<decltype(in[0])>, FloatType>::value, "Input vector should be the same type as this structure.");
        static_assert(is_same<decay_t<decltype(out[0])>, FloatType>::value, "Output vector should be the same type as this structure.");
        out = in;
        apply(out);
    }
    template<typename FloatType2>
    void apply(FloatType2 *vec) {
        auto tmp(as_type<FloatType2>());
        for(T i = 0; i < size(); ++i) vec[i] *= tmp[i];
    }
    template<typename VectorType>
    void apply(VectorType &vec) {
        auto tmp(as_type<std::decay_t<decltype(vec[0])>>());
        if(vec.size() != size()) {
            if(vec.size() > size())
                throw std::runtime_error("Vector is too big for he gotdam feet");
            std::fprintf(stderr, "Warning: CompactRademacherTemplate is too small. Only affecting elements in my size.\n");
        }
#if USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 8192)
#endif
        // Think about loading words and working my way manually.
        for(T i = 0; i < size(); ++i) {
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
