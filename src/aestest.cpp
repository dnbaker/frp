#include <string>
#include <random>
#include <iostream>
#include <type_traits>
#include <vector>
#include <chrono>
#include "gfrp/gfrp.h"

class Timer {
    using TpType = std::chrono::system_clock::time_point;
    std::string name_;
    TpType start_, stop_;
public:
    Timer(std::string &&name=""): name_{name}, start_(std::chrono::system_clock::now()) {}
    void stop() {stop_ = std::chrono::system_clock::now();}
    void restart() {start_ = std::chrono::system_clock::now();}
    void report() {std::cerr << "Took " << std::chrono::duration<double>(stop_ - start_).count() << "s for task '" << name_ << "'\n";}
    ~Timer() {stop(); /* hammertime */ report();}
};

template<typename Dist, typename RNG>
auto make_val(Dist &dist, RNG &rng) {
    return dist(rng);
}

template<typename Dist, typename RNG, typename FloatType>
void time_stuff(const char *name, Dist &dist, RNG &rng, std::vector<FloatType> &vals, size_t niter, size_t size) {
    Timer t(name);
    while(niter--) for(size_t i(0); i < size; ++i) vals[i] = dist(rng);
}

template<typename RNG>
void time_aes(RNG &rng, std::vector<uint64_t> &vec, std::string name, size_t nrounds) {
    {
        Timer t(name + "sequential");
        for(size_t i(0); i < nrounds; ++i)
            for(size_t j(0); j < vec.size(); ++j) 
                vec[i] = rng();
    }
    {
        Timer t(name + "ram");
        for(size_t i(0); i < nrounds; ++i)
            for(size_t j(0); j < vec.size(); ++j) 
                vec[i] = rng[i];
    }
}

template<typename... Types>
using unormd = boost::random::detail::unit_normal_distribution<Types...>;

int main(int argc, char *argv[]) {
    size_t niter(argc > 2 ? std::atoi(argv[2]): 1000), size(argc > 1 ? std::atoi(argv[1]): 1 << 16);
    int64_t UNROLL_COUNT = 4;
    const __m128i CTR_ADD = {UNROLL_COUNT, 0};
    const __m128i CTR_CMP = _mm_set_epi64x(0, UNROLL_COUNT);
    assert(CTR_ADD[0] == CTR_CMP[0]);
    assert(CTR_ADD[1] == CTR_CMP[1]);
    std::vector<uint64_t> vec(size);
    aes::AesCtr<uint64_t, 8> c8;
    aes::AesCtr<uint64_t, 4> c4;
    aes::AesCtr<uint64_t, 16> c16;
    aes::AesCtr<uint64_t, 32> c32;
    aes::AesCtr<uint64_t, 64> c64;
    aes::AesCtr<uint64_t, 128> c128;
    aes::AesCtr<uint32_t, 16> c8_16;
    time_aes(c8, vec, "c8", niter);
    time_aes(c16, vec, "c16", niter);
    std::cerr << "sizeof aesctr: " << sizeof(c8) << '\n';
    std::vector<float> vals(size);
    std::vector<double> dvals(size);
    unormd<float> bnd;
    std::normal_distribution<float>   snd;
    unormd<double> dbnd;
    std::normal_distribution<double>   dsnd;
    time_stuff("bnd", bnd, c8, vals, niter, size);
    time_stuff("snd", snd, c8, vals, niter, size);
    time_stuff("dbnd", dbnd, c8, dvals, niter, size);
    time_stuff("snd", dsnd, c8, dvals, niter, size);
    time_stuff("u32dbnd", dbnd, c8_16, dvals, niter, size);
    time_stuff("u32dsnd", dsnd, c8_16, dvals, niter, size);
    gfrp::OnlineShuffler shuf;
    gfrp::OJLTransform<3> t(size, size / 8, 1440);
#if 0
    aes::AesCtr test(1337);
    for(size_t i(0); i < 1000; ++i)
        std::cerr << test() << '\n';
#endif
}
