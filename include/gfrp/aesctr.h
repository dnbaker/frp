#ifndef AESCTR_H
#define AESCTR_H

// Taken from https://github.com/lemire/testingRNG
// Added C++ interface compatible with std::shuffle, &c.

// contributed by Samuel Neves

#include <cassert>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <immintrin.h>

namespace aes {

using std::uint64_t;
using std::uint8_t;
using std::size_t;
using std::numeric_limits;



#define AES_ROUND(rcon, index)                                                 \
  do {                                                                         \
    __m128i k2 = _mm_aeskeygenassist_si128(k, rcon);                           \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3, 3, 3, 3)));      \
    seed_[index] = k;                                                    \
  } while (0)


template<typename T>
struct is_aes: std::false_type {};

template<typename GeneratedType=uint64_t, size_t UNROLL_COUNT=8>
class AesCtr {
    static const size_t AESCTR_ROUNDS = 10;
    uint8_t state_[sizeof(__m128i) * UNROLL_COUNT];
    __m128i ctr_[UNROLL_COUNT];
    __m128i seed_[AESCTR_ROUNDS + 1];
    __m128i work_[UNROLL_COUNT];
    size_t offset_;

    // Unrollers
    template<size_t ind, size_t todo>
    struct aes_unroll_impl {
        void operator()(__m128i *ret, AesCtr &state) const {
            ret[ind] = _mm_xor_si128(state.ctr_[ind], state.seed_[0]);
            aes_unroll_impl<ind + 1, todo - 1>()(ret, state);
        }
        void aesenc(__m128i *ret, __m128i subkey) const {
            ret[ind] = _mm_aesenc_si128(ret[ind], subkey);
            aes_unroll_impl<ind + 1, todo - 1>().aesenc(ret, subkey);
        }
        template<size_t NUMROLL>
        void round_and_enc(__m128i *ret, AesCtr &state) const {
            const __m128i subkey = state.seed_[ind];
            aes_unroll_impl<0, NUMROLL>().aesenc(ret, subkey);
            aes_unroll_impl<ind + 1, todo - 1>().template round_and_enc<NUMROLL>(ret, state);
        }
        void add_store(__m128i *work, AesCtr &state) const {
            static_assert(sizeof(__m128i) == 16, "Pretty sure this has to be true.");
            static const __m128i CTR_ADD =  _mm_set_epi64x(0, UNROLL_COUNT);
            state.ctr_[ind] =
                _mm_add_epi64(state.ctr_[ind], CTR_ADD);
                _mm_storeu_si128(
                    (__m128i *)&state.state_[sizeof(__m128i) * ind],
                    _mm_aesenclast_si128(work[ind], state.seed_[AESCTR_ROUNDS]));
            aes_unroll_impl<ind + 1, todo - 1>().add_store(work, state);
        }
        template<typename T>
        void pluseq(T *data, T addn) {
#if __AVX2__
            data[ind] = _mm256_add_epi64(data[ind], addn);
#else
            data[ind] = _mm_add_epi64(data[ind], addn);
#endif
            aes_unroll_impl<ind + 1, todo - 1>()(data, addn);
        }
    };
    // Termination conditions
    template<size_t ind>
    struct aes_unroll_impl<ind, 0> {
        void operator()([[maybe_unused]] __m128i *ret, [[maybe_unused]] AesCtr &state) const {}
        void aesenc([[maybe_unused]] __m128i *ret, [[maybe_unused]] __m128i subkey) const {}
        template<size_t NUMROLL>
        void round_and_enc([[maybe_unused]] __m128i *ret, [[maybe_unused]] AesCtr &state) const {}
        void add_store([[maybe_unused]] __m128i *work, [[maybe_unused]] AesCtr &state) const {}
        void pluseq([[maybe_unused]] __m128i *work, [[maybe_unused]] __m128i addn) const {}
    };

public:
    using result_type = GeneratedType;
    AesCtr(uint64_t seedval=0) {
        seed(seedval);
    }
    result_type operator()() {
        if (__builtin_expect(offset_ >= sizeof(__m128i) * UNROLL_COUNT, 0)) {
            aes_unroll_impl<0, UNROLL_COUNT>()(work_, *this);
            aes_unroll_impl<1, AESCTR_ROUNDS - 1>().template round_and_enc<UNROLL_COUNT>(work_, *this);
            aes_unroll_impl<0, UNROLL_COUNT>().add_store(work_, *this);
            offset_ = 0;
        }
        result_type ret;
        memcpy(&ret, state_ + offset_, sizeof(ret));
        offset_ += sizeof(result_type);
        return ret;
    }
    void fast_forward(uint64_t index) {
        static const uint64_t MASK = (sizeof(__m128i) * UNROLL_COUNT / sizeof(GeneratedType) - 1);
        static_assert((MASK & (MASK + 1)) == 0, "I MUST HAVE THIS NOT WRONG");
        const uint64_t chunkstart(index & ~(MASK));
        std::cerr << "For index " << index << " I have chunkstart " << chunkstart << " And leftover " << (index & MASK) << '\n';
        assert((chunkstart & (MASK)) == 0);
        for(uint64_t i(0); i < UNROLL_COUNT; ++i) {
            ctr_[i] = _mm_set_epi64x(0, chunkstart + i);
        }
        aes_unroll_impl<0, UNROLL_COUNT>()(work_, *this);
        aes_unroll_impl<1, AESCTR_ROUNDS - 1>().template round_and_enc<UNROLL_COUNT>(work_, *this);
        for(uint64_t i(0); i < UNROLL_COUNT; ++i) {
            _mm_storeu_si128(
                (__m128i *)&state_[sizeof(__m128i) * i],
                _mm_aesenclast_si128(work_[i], seed_[AESCTR_ROUNDS]));
        }
        offset_ = index & (sizeof(__m128i) * UNROLL_COUNT / sizeof(GeneratedType) - 1);
    }
    static constexpr result_type max() {return numeric_limits<result_type>::max();}
    static constexpr result_type min() {return numeric_limits<result_type>::min();}
    void seed(uint64_t k) {
        seed(_mm_set_epi64x(0, k));
    }
    void seed(__m128i k) {
      seed_[0] = k;
      // D. Lemire manually unrolled following loop since _mm_aeskeygenassist_si128
      // requires immediates

      AES_ROUND(0x01, 1);
      AES_ROUND(0x02, 2);
      AES_ROUND(0x04, 3);
      AES_ROUND(0x08, 4);
      AES_ROUND(0x10, 5);
      AES_ROUND(0x20, 6);
      AES_ROUND(0x40, 7);
      AES_ROUND(0x80, 8);
      AES_ROUND(0x1b, 9);
      AES_ROUND(0x36, 10);

      for (unsigned i = 0; i < UNROLL_COUNT; ++i) ctr_[i] = _mm_set_epi64x(0, i);
      offset_ = sizeof(__m128i) * UNROLL_COUNT;
    }
#if 1
    result_type operator[](size_t count) const {
        static constexpr unsigned DIV   = sizeof(__m128i) / sizeof(result_type);
        static constexpr unsigned BMASK = DIV - 1;
        const unsigned offset_(count & BMASK);
        result_type ret[DIV];
        count /= DIV;
        //std::fprintf(stderr, "Count: %zu. DIV: %zu. BMASK: %zu\n", count, DIV, BMASK);
        __m128i tmp(_mm_xor_si128(_mm_set_epi64x(0, count), seed_[0]));
        for (unsigned r = 1; r <= AESCTR_ROUNDS - 1; ++r) {
            tmp = _mm_aesenc_si128(tmp, seed_[r]);
        }
        _mm_storeu_si128((__m128i *)ret, _mm_aesenclast_si128(tmp, seed_[AESCTR_ROUNDS]));
        //std::cerr << "About to return value " << ret[offset_] << '\n';
        return ret[offset_];
#else
    result_type operator[](size_t count) const {
        fast_forward(count);
        static constexpr unsigned DIV   = sizeof(__m128i) / sizeof(result_type);
        static constexpr unsigned BMASK = DIV - 1;
        const unsigned offset_(count & BMASK);
        result_type ret[DIV];
        count /= DIV;
        std::fprintf(stderr, "Count: %zu. DIV: %zu. BMASK: %zu\n", count, DIV, BMASK);
        __m128i tmp(_mm_xor_si128(_mm_set_epi64x(0, count), seed_[0]));
        for (unsigned r = 1; r <= AESCTR_ROUNDS - 1; ++r) {
            tmp = _mm_aesenc_si128(tmp, seed_[r]);
        }
        _mm_storeu_si128((__m128i *)ret, _mm_aesenclast_si128(tmp, seed_[AESCTR_ROUNDS]));
        std::cerr << "About to return value " << ret[offset_] << '\n';
        return ret[offset_];
#endif
    }
};
#undef AES_ROUND

template<typename T, size_t n>
struct is_aes<AesCtr<T, n>>: std::true_type {};


} // namespace aes

#undef AESCTR_UNROLL
#undef AESCTR_ROUNDS

#endif
