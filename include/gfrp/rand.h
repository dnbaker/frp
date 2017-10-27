#ifndef _GRFP_RAND_H__
#define _GRFP_RAND_H__
#include <random>
#include "fastrange/fastrange.h"
#include "include/thirdparty/fast_mutex.h"
#include "include/thirdparty/aesctr.h"

namespace rng {

struct RandTwister {

    std::mt19937_64 twister_;

    using ResultType = std::mt19937_64::result_type;

    static const ResultType MAX     = std::mt19937_64::max();
    static const ResultType MIN     = std::mt19937_64::min();
    static constexpr double MAX_INV = 1. / MAX;

    RandTwister(ResultType seed=std::rand()): twister_(seed) {}
    void seed(ResultType seed) {twister_.seed(seed);}
    auto operator()()                              {return twister_();}
    auto operator()(std::mt19937_64 &engine) const {return engine();}
    // Generate a large number of random integers.
    template<typename IntType>
    void operator()(IntType n, RandTwister::ResultType *a) {
#if UNROLL
        const auto leftover(n & 0x7UL);
        n >>= 3;
        switch(leftover) {
            case 0: do {
                    *a++ = twister_();
            case 7: *a++ = twister_();
            case 6: *a++ = twister_();
            case 5: *a++ = twister_();
            case 4: *a++ = twister_();
            case 3: *a++ = twister_();
            case 2: *a++ = twister_();
            case 1: *a++ = twister_();
                       } while(n--);
        }
#else
        for(;n--;*a++ = twister_());
#endif
    }
    void reseed(ResultType seed) {twister_.seed(seed);}
    using TwisterReference = std::add_lvalue_reference<std::mt19937_64>::type;
    operator TwisterReference() {
        return twister_;
    }
};

struct ThreadsafeRandTwister: public RandTwister {
    ThreadsafeRandTwister(ResultType seed): RandTwister(seed) {}
    tthread::fast_mutex lock_;
    void seed(ResultType seed) {lock_.lock(); RandTwister::seed(seed); lock_.unlock();}
    auto operator()() {
        lock_.lock();
        const auto ret(twister_());
        lock_.unlock();
        return ret;
    }
    // Generate a large number of random integers.
    auto operator()(std::size_t n, ResultType *a) {
        lock_.lock();
        RandTwister::operator ()(n, a);
        lock_.unlock();
    }
};

static RandTwister random_twist(std::rand());
static ThreadsafeRandTwister tsrandom_twist(std::rand());

// Based on https://github.com/lemire/FastShuffleExperiments/blob/master/cpp/rangedrand.h
// map random value to [0,range) with slight bias, redraws to avoid bias if
// needed
template <typename RandomBitGenerator>
static inline uint64_t random_bounded_nearlydivisionless64(uint64_t range, RandomBitGenerator &rg) {
  __uint128_t random64bit, multiresult;
  uint64_t leftover;
  uint64_t threshold;
  random64bit = rg();
  multiresult = random64bit * range;
  leftover = (uint64_t)multiresult;
  if (leftover < range) {
    threshold = -range % range;
    while (leftover < threshold) {
      random64bit = rg();
      multiresult = random64bit * range;
      leftover = (uint64_t)multiresult;
    }
  }
  return multiresult >> 64; // [0, range)
}

static inline uint64_t random_bounded_nearlydivisionless64(uint64_t range) {
    return random_bounded_nearlydivisionless64<rng::RandTwister>(range, random_twist);
}

static inline uint64_t tsrandom_bounded_nearlydivisionless64(uint64_t range) {
    return random_bounded_nearlydivisionless64<rng::ThreadsafeRandTwister>(range, tsrandom_twist);
}

template<typename T>
T randf() {return random_twist() * RandTwister::MAX_INV;}
template<typename T>
T tsrandf() {return tsrandom_twist() * RandTwister::MAX_INV;}

} // namespace rng

#endif // #ifndef _GRFP_RAND_H__
