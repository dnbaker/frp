#ifndef AESCTR_H
#define AESCTR_H

// Taken from https://github.com/lemire/testingRNG
// Added C++ interface compatible with std::shuffle, &c.

// contributed by Samuel Neves

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


template<typename GeneratedType=uint64_t, size_t UNROLL_COUNT=8>
class AesCtr {
    static const size_t AESCTR_ROUNDS = 10;
    uint8_t state_[sizeof(__m128i) * UNROLL_COUNT];
    __m128i ctr_[UNROLL_COUNT];
    __m128i seed_[AESCTR_ROUNDS + 1];
    __m128i work[UNROLL_COUNT];
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
          state.ctr_[ind] =
              _mm_add_epi64(state.ctr_[ind], _mm_set_epi64x(0, UNROLL_COUNT));
              _mm_storeu_si128(
                  (__m128i *)&state.state_[16 * ind],
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
            aes_unroll_impl<0, UNROLL_COUNT>()(work, *this);
            aes_unroll_impl<1, AESCTR_ROUNDS - 1>().template round_and_enc<UNROLL_COUNT>(work, *this);
            aes_unroll_impl<0, UNROLL_COUNT>().add_store(work, *this);
            offset_ = 0;
        }
        result_type ret;
        memcpy(&ret, state_ + offset_, sizeof(ret));
        offset_ += sizeof(result_type);
        return ret;
    }
#if 0
    // This code is wrong but might be on the right track.
    void fast_forward(uint64_t index) {
        const uint64_t uval(index & ~(UNROLL_COUNT - 1));
        if(uval) {
#if __AVX2__
            const __m256i new_offset = _mm256_set_epi64x(0, uval, 0, uval);
            aes_unroll_impl<0, UNROLL_COUNT >> 1>().pluseq((__m256i *)ctr_, new_offset);
            if constexpr(UNROLL_COUNT & 1) {
                ctr_[UNROLL_COUNT - 1] = _mm_add_epi64(ctr_[UNROLL_COUNT - 1], _mm_set_epi64x(0, uval));
            }
#else
            const __m128i new_offset = _mm_set_epi64x(0, uval);
            aes_unroll_impl<0, UNROLL_COUNT>().pluseq(ctr_, new_offset);
        }
#endif
        offset_ = (index - uval) * sizeof(__m128i);
    }
#endif
    result_type max() const {return numeric_limits<result_type>::max();}
    result_type min() const {return numeric_limits<result_type>::min();}
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
    result_type operator[](size_t count) const {
        static constexpr unsigned DIV   = sizeof(__m128i) / sizeof(result_type);
        static constexpr unsigned BMASK = DIV - 1;
        const unsigned offset_(count & BMASK);
        result_type ret[DIV];
        count /= DIV;
        __m128i tmp(_mm_xor_si128(_mm_set_epi64x(0, count), seed_[0]));
        for (unsigned r = 1; r <= AESCTR_ROUNDS - 1; ++r) {
            tmp = _mm_aesenc_si128(tmp, seed_[r]);
        }
        _mm_storeu_si128((__m128i *)ret, _mm_aesenclast_si128(tmp, seed_[AESCTR_ROUNDS]));
        return ret[offset_];
    }
};
#undef AES_ROUND

} // namespace aes

#undef AESCTR_UNROLL
#undef AESCTR_ROUNDS

#endif
