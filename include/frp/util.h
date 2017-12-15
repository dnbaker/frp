#ifndef _GFRP_UTIL_H__
#define _GFRP_UTIL_H__
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <memory>
#include <tuple>
#include <type_traits>
#include <type_traits>
#include <unordered_set>
#include "kspp/ks.h"
#include "FFHT/fast_copy.h"
#include "blaze/Math.h"
#include "boost/random/normal_distribution.hpp"
#include "boost/random.hpp"

#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif

#ifdef __GNUC__
#  ifndef likely
#    define likely(x) __builtin_expect((x),1)
#  endif
#  ifndef unlikely
#    define unlikely(x) __builtin_expect((x),0)
#  endif
#  ifndef UNUSED
#    define UNUSED(x) __attribute__((unused)) x
#  endif
#else
#  ifndef likely
#    define likely(x) (x)
#  endif
#  ifndef unlikely
#    define unlikely(x) (x)
#  endif
#  ifndef UNUSED
#    define UNUSED(x) (x)
#  endif
#endif

#ifndef IS_BLAZE
#define IS_BLAZE(x) (blaze::IsVector<x>::value || blaze::IsMatrix<x>::value)
#endif
#ifndef IS_COMPRESSED_BLAZE
#define IS_COMPRESSED_BLAZE(x) (blaze::IsSparseVector<x>::value || blaze::IsSparseMatrix<x>::value)
#endif
#ifndef IS_UNCOMPRESSED_BLAZE
#define IS_UNCOMPRESSED_BLAZE(x) (IS_BLAZE(x) && !IS_COMPRESSED_BLAZE(x))
#endif

namespace frp {
using namespace std::literals;
using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::int64_t;
using std::int32_t;
using std::int16_t;
using std::int8_t;
using blaze::DynamicVector;
using blaze::DynamicMatrix;
using blaze::TransposeFlag;
using std::size_t;
using std::enable_if_t;
using std::decay_t;
using std::memset;
using std::memcpy;
using std::malloc;
using std::realloc;
using std::unique_ptr;
using std::is_arithmetic;
using std::is_floating_point;
using std::runtime_error;
using std::bad_alloc;
using std::unordered_set;
using std::forward;
using std::is_same;
using std::FILE;
using std::fprintf;
using std::sprintf;
using std::numeric_limits;
using std::strstr;
using std::atoi;
using std::fclose;
using std::exit;
using std::cerr;
using std::cout;
using u32 = uint32_t;
using u64 = uint64_t;

inline constexpr uint64_t roundup(uint64_t x) {
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return ++x;
}
template < template <typename...> class Template, typename T >
struct is_instantiation_of : std::false_type {};

template < template <typename...> class Template, typename... Args >
struct is_instantiation_of< Template, Template<Args...> > : std::true_type {};

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
    void rename(const char *name) {name_ = name;}
};

inline constexpr int log2_64(uint64_t value)
{
    // https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
    const int tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5
    };
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
}

template<typename T> class TD;

template<class Container>
auto normsq(const Container &c) {
    return dot(c, c);
}
template<class Container>
auto norm(const Container &c) {
    return std::sqrt(dot(c, c));
}

template<class Container>
auto mean(const Container &c) {
    using FloatType = decay_t<decltype(c[0])>;
    FloatType sum(0.);
    if constexpr(blaze::IsSparseVector<Container>::value || blaze::IsSparseVector<Container>::value) {
        for(const auto entry: c) sum += entry.value();
    } else {
        for(const auto entry: c) sum += entry;
    }
    sum /= c.size();
    return sum;
}

template<class Container>
auto sum(const Container &c) {
    return std::accumulate(c.begin(), c.end(), static_cast<decay_t<decltype(*c.begin())>>(0));
}

template<class Container>
auto meanvar(const Container &c) {
    using FloatType = decay_t<decltype(c[0])>;
    FloatType sum(0.), varsum(0.0);
    if constexpr(blaze::IsSparseVector<Container>::value || blaze::IsSparseVector<Container>::value) {
        for(const auto entry: c) sum += entry.value(), varsum += entry.value() * entry.value();
    } else {
        for(const auto entry: c) sum += entry, varsum += entry * entry;
    }
    const auto inv(static_cast<FloatType>(1)/static_cast<FloatType>(c.size()));
    varsum -= sum * sum * inv;
    varsum *= inv;
    sum *= inv;
    return std::make_pair(sum, varsum);
}

template<typename T>
void ksprint(const T &view, ks::string &buf, bool scientific=true) {
    const char fmt[5] = {'%', 'l', (char)('f' - scientific), ',', '\0'};
    for(const auto el: view) buf.sprintf(fmt, static_cast<double>(el));
    buf.pop();
}

template<typename T>
void pv(const T &view, FILE *fp=stderr) {
    ks::string str;
    ksprint(view, str);
    str.terminate();
    std::cerr << str.data();
}


size_t countchars(const char *line, int delim) {
    size_t ret(0);
    while(*line && *line != '\n') ret += (*line++ == delim);
    return ret;
}

} // namespace frp

#endif
