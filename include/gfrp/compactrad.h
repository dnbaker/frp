#ifndef _GFRP_CRAD_H__
#define _GFRP_CRAD_H__
#include "gfrp/util.h"
#include "gfrp/linalg.h"

namespace gfrp { namespace dist {

template<typename T=uint64_t, typename FloatType=FLOAT_TYPE, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
class CompactRademacher {
    size_t n_, m_;
    T *data_;
    const FloatType values_[2];
    static constexpr size_t NBITS = sizeof(T) * CHAR_BIT;
    static constexpr size_t SHIFT = log2_64(NBITS);
    static constexpr size_t BITMASK = NBITS - 1;

    using value_type = FloatType;
    using container_type = T;
    using size_type = size_t;
public:
    CompactRademacher(size_t n=0): n_{n}, m_{n}, data_(static_cast<T *>(std::malloc((sizeof(T) * n) >> SHIFT))), values_{-1, 1} {
        if(n & (n - 1)) {
            std::fprintf(stderr, "Warning: n is not a power of 2. This is a surprise. (%zu)\n", n);
        }
    }
    // For setting to random values
    auto *data() {return data_;}
    const auto *data() const {return data_;}
    // For use
    auto size() const {return n_ << SHIFT;}
    auto capacity() const {return m_ << SHIFT;}
    template<typename OWordType, typename OFloatType>
    bool operator==(const CompactRademacher<OWordType, OFloatType> &other) const {
        if(size() != other.size()) return false;
        auto odata = other.data();
        for(size_t i(0);i < n_; ++i)
            if(data_[i] != odata[i])
                return false;
        return true;
    }
    void zero() {std::memset(data_, 0, sizeof(T) * (n_ >> SHIFT));}
    void reserve(size_t newsize) {
        if(newsize & (newsize - 1)) throw std::runtime_error("newsize should be a power of two");
        if(newsize > m_) {
            auto tmp(static_cast<T*>(std::realloc(data_, sizeof(T) * (newsize >> SHIFT))));
            if(tmp == nullptr) throw std::bad_alloc();
            data_ = tmp;
        }
    }
    FloatType operator[](size_type idx) const {return values_[!!(data_[(idx >> SHIFT)] & (static_cast<T>(1) << (idx & BITMASK)))] ;}
    ~CompactRademacher(){
        std::free(data_);
    }
};


}}


#endif // #ifndef _GFRP_CRAD_H__
