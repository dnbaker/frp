#ifndef AESCTR_H
#define AESCTR_H

// Taken from https://github.com/lemire/testingRNG
// Added C++ interface compatible with std::shuffle, &c.

// contributed by Samuel Neves

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace aes {
using std::uint64_t;
using std::uint8_t;
using std::size_t;

#ifndef AESCTR_UNROLL
#define AESCTR_UNROLL 4
#endif
#ifndef AESCTR_ROUNDS
#define AESCTR_ROUNDS 10
#endif


// TODO: Fully template this.
typedef struct {
  uint8_t state[16 * AESCTR_UNROLL];
  __m128i ctr[AESCTR_UNROLL];
  __m128i seed[AESCTR_ROUNDS + 1];
  size_t offset;
} aesctr_state;

#define AES_ROUND(rcon, index)                                                 \
  do {                                                                         \
    __m128i k2 = _mm_aeskeygenassist_si128(k, rcon);                           \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3, 3, 3, 3)));      \
    state->seed[index] = k;                                                    \
  } while (0)

static inline void aesctr_seed_r(aesctr_state *state, __m128i k) {
  /*static const uint8_t rcon[] = {
      0x8d, 0x01, 0x02, 0x04,
      0x08, 0x10, 0x20, 0x40,
      0x80, 0x1b, 0x36
  };*/
  state->seed[0] = k;
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

  for (int i = 0; i < AESCTR_UNROLL; ++i) {
    state->ctr[i] = _mm_set_epi64x(0, i);
  }
  state->offset = sizeof(__m128i) * AESCTR_UNROLL;
}

static inline void aesctr_seed_r(aesctr_state *state, uint64_t seed) {
    aesctr_seed_r(state, _mm_set_epi64x(0, seed));
}

#undef AES_ROUND

template<size_t ind, size_t todo>
struct aes_unroll_impl {
    operator()(__m128i *ret, aesctr_state *state) const {
        ret[ind] = _mm_xor_si128(state->ctr[ind], state->seed[0]);
        aes_unroll_impl<ind + 1, todo - 1>()(ret, state);
    }
    void aesenc(__m128i *ret, __m128i subkey) const {
        ret[ind] = _mm_aesenc_si128(ret[ind], subkey);
        aes_unroll_impl<ind + 1, todo - 1>().aesenc(ret, subkey);
    }
    template<size_t NUMROLL>
    void round_and_enc(__m128i *ret, aesctr_state *state) const {
        const __m128i subkey = state->seed[ind];
        aes_unroll_impl<0, NUMROLL>().aesenc(ret, subkey);
        aes_unroll_impl<ind + 1, todo - 1>().template round_and_enc<NUMROLL>(ret, state);
    }
    void add_store(__m128i *work, aesctr_state *state) const {
      state->ctr[ind] =
          _mm_add_epi64(state->ctr[ind], _mm_set_epi64x(0, AESCTR_UNROLL));
          _mm_storeu_si128(
              (__m128i *)&state->state[16 * ind],
              _mm_aesenclast_si128(work[ind], state->seed[AESCTR_ROUNDS]));
      aes_unroll_impl<ind + 1, todo - 1>().add_store(work, state);
    }
};

template<size_t ind>
struct aes_unroll_impl<ind, 0> {
    operator()(__m128i *ret, aesctr_state *state) const {}
    void aesenc(__m128i *ret, __m128i subkey) const {}
    template<size_t NUMROLL>
    void round_and_enc(__m128i *ret, aesctr_state *state) const {}
    void add_store(__m128i *work, aesctr_state *state) const {}
};

template<typename T, size_t aes_unroll_ct=AESCTR_UNROLL,
         typename=std::enable_if_t<std::is_integral<T>::value>>
static inline T aesctr_r(aesctr_state *state) {
  if (__builtin_expect(state->offset >= sizeof(__m128i) * AESCTR_UNROLL, 0)) {
    __m128i work[aes_unroll_ct];
    aes_unroll_impl<0, aes_unroll_ct>()(work, state);
    aes_unroll_impl<1, AESCTR_ROUNDS - 1>().template round_and_enc<aes_unroll_ct>(work, state);
    aes_unroll_impl<0, aes_unroll_ct>().add_store(work, state);
    state->offset = 0;
  }
  T output;
  memcpy(&output, &state->state[state->offset], sizeof(output));
  state->offset += sizeof(output);
  return output;
}

template<typename T, typename=std::enable_if_t<std::is_integral<T>::value>>
static inline T aes_random_access_r(const aesctr_state *state, size_t count) {
    // Since AES generates 64-bit values, we have to select one of two results.
    static constexpr unsigned DIV   = sizeof(__m128i) / sizeof(T);
    static constexpr unsigned BMASK = DIV - 1;
    const unsigned offset(count & BMASK);
    T ret[DIV];
    count /= DIV;
    __m128i work(_mm_xor_si128(_mm_set_epi64x(0, count), state->seed[0]));
    for (int r = 1; r <= AESCTR_ROUNDS - 1; ++r) {
        work = _mm_aesenc_si128(work, state->seed[r]);
    }
    _mm_storeu_si128((__m128i *)ret, _mm_aesenclast_si128(work, state->seed[AESCTR_ROUNDS]));
    return ret[offset];
}

static aesctr_state g_aesctr_state;

static inline void aesctr_seed(uint64_t seed) {
  aesctr_seed_r(&g_aesctr_state, seed);
}

static inline uint64_t aesctr() { return aesctr_r<uint64_t>(&g_aesctr_state); }

template<typename GeneratedType=uint64_t, size_t unroll_size=4, typename=std::enable_if_t<std::is_integral<GeneratedType>::value>>
class AesCtr {
    // Todo: template this to provide random bits of various sizes.
    aesctr_state ctr_;
public:
    using result_type = GeneratedType;
    AesCtr(uint64_t seed=0): ctr_{{0}, {0}, {0}, 0} {
        aesctr_seed_r(&ctr_, seed);
    }
    result_type operator()() {
        return aesctr_r<result_type, unroll_size>(&ctr_);
    }
    void seed(uint64_t seedval) {
        aesctr_seed_r(&ctr_, seedval);
    }
    result_type operator[] (size_t index) const {
        return aes_random_access_r<result_type>(&ctr_, index);
    }
    result_type max() const {return std::numeric_limits<uint64_t>::max();}
    result_type min() const {return std::numeric_limits<uint64_t>::min();}
};

} // namespace aes

#undef AESCTR_UNROLL
#undef AESCTR_ROUNDS

#endif
