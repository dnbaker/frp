#include <string>
#include <random>
#include <iostream>
#include <type_traits>
#include <vector>
#include <chrono>
#include "frp/frp.h"

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
    size_t niter(argc > 2 ? std::strtoull(argv[2], 0, 10): 1000), size(argc > 1 ? std::strtoull(argv[1], 0, 10): 1 << 16);
    std::fprintf(stderr, "niter %zu size %zu\n", niter, size);
    int64_t UNROLL_COUNT = 4;
    const __m128i CTR_ADD = {UNROLL_COUNT, 0};
    const __m128i CTR_CMP = _mm_set_epi64x(0, UNROLL_COUNT);
    assert(CTR_ADD[0] == CTR_CMP[0]);
    assert(CTR_ADD[1] == CTR_CMP[1]);
    std::vector<uint64_t> vec(size);
    aes::AesCtr<uint64_t, 8> c8;
    aes::AesCtr<uint64_t, 4> c4;
    std::vector<double> rvals(size);
    std::vector<float> fvals(size);
    std::uniform_real_distribution<double> urdd(0, M_PI * 2);
    std::uniform_real_distribution<double> urdf(0, M_PI * 2);
    boost::random::uniform_real_distribution<double> burdd(0, M_PI * 2);
    boost::random::uniform_real_distribution<double> burdf(0, M_PI * 2);
    {
        Timer t("rdd");
        for(size_t i(0); i < niter; ++i)
        for(auto &el: rvals) el = urdd(c8);
    }
    {
        Timer t("rdf");
        for(size_t i(0); i < niter; ++i)
        for(auto &el: fvals) el = urdf(c8);
    }
    {
        Timer t("rdfd");
        for(size_t i(0); i < niter; ++i)
        for(auto &el: fvals) el = urdd(c8);
    }
    {
        Timer t("rdd");
        for(size_t i(0); i < niter; ++i)
        for(auto &el: rvals) el = burdd(c8);
    }
    {
        Timer t("rdf");
        for(size_t i(0); i < niter; ++i)
        for(auto &el: fvals) el = burdf(c8);
    }
    {
        Timer t("rdfd");
        for(size_t i(0); i < niter; ++i)
        for(auto &el: fvals) el = burdd(c8);
    }
    blaze::DynamicVector<float> zomgz(1 << 8);
    for(size_t i(0); i < zomgz.size(); zomgz[i] = i, ++i);
    frp::LutShuffler<size_t> os(zomgz.size(), 1);
    os.apply(zomgz);
    std::cerr << zomgz;
}
