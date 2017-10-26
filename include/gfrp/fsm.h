#ifndef _GFRP_FSM_H__
#define _GFRP_FSM_H__

namespace fsm { // Fast structured multiplication
using std::uint64_t;
using std::uint32_t;

template<size_t POW> constexpr uint64_t POWER_2 = 1ull << POW;
static const uint64_t POW2_LUT [] {
    POWER_2<0>, POWER_2<1>, POWER_2<2>, POWER_2<3>, POWER_2<4>, POWER_2<5>, POWER_2<6>, POWER_2<7>, POWER_2<8>, POWER_2<9>, POWER_2<10>, POWER_2<11>, POWER_2<12>, POWER_2<13>, POWER_2<14>, POWER_2<15>, POWER_2<16>, POWER_2<17>, POWER_2<18>, POWER_2<19>, POWER_2<20>, POWER_2<21>, POWER_2<22>, POWER_2<23>, POWER_2<24>, POWER_2<25>, POWER_2<26>, POWER_2<27>, POWER_2<28>, POWER_2<29>, POWER_2<30>, POWER_2<31>, POWER_2<32>, POWER_2<33>, POWER_2<34>, POWER_2<35>, POWER_2<36>, POWER_2<37>, POWER_2<38>, POWER_2<39>, POWER_2<40>, POWER_2<41>, POWER_2<42>, POWER_2<43>, POWER_2<44>, POWER_2<45>, POWER_2<46>, POWER_2<47>, POWER_2<48>, POWER_2<49>, POWER_2<50>, POWER_2<51>, POWER_2<52>, POWER_2<53>, POWER_2<54>, POWER_2<55>, POWER_2<56>, POWER_2<57>, POWER_2<58>, POWER_2<59>, POWER_2<60>, POWER_2<61>, POWER_2<62>, POWER_2<63>
};

#define _pow2(x) (POW2_LUT[x])

// Dumb fht
// Modified from https://github.com/FALCONN-LIB/FFHT
template<typename Source, typename DestType>
void rad_fht(const Source &in, DestType out, uint64_t log_n) {
    static_assert(std::is_same<std::decay_t<decltype(in[0])>, std::decay_t<decltype(out[0])>>::value, "Source must dereference to the same value as destination.");
    using FloatType = std::decay_t<decltype(out[0])>;
    const uint64_t n(_pow2(log_n));
    uint64_t i, j, k;
    FloatType tmp1, tmp2;
    uint64_t ti1, ti2, s1, s2;
    for (uint64_t j = 0; j < n; j += 2) {
        tmp1 = in[j], tmp2 = in[j+1];
        out[j] = tmp1 + tmp2;
        out[j+1] = tmp1 - tmp2;
    }
    for (i = 1, s1 = 1, s2 = 2; i < log_n; ++i) {
        for (j = 0; j < n; j += s2) {
            for (k = 0; k < s1; ++k) {
                ti1 = j + k, ti2 = ti1 + s1;
                tmp1 = out[ti1], tmp2 = out[ti2];
                out[ti1] = tmp1 + tmp2;
                out[ti2] = tmp1 - tmp2;
            }
        }
        s1 <<= 1, s2 <<= 1;
    }
}

#undef _pow2
template<typename FloatType>
void dumb_fht(FloatType *buf, const uint64_t log_n) {
    uint64_t n = 1ull << log_n, s1 = 1, s2 = 2;
    FloatType u, v;
    for (uint64_t i = 0; i < log_n; ++i) {
        for (uint64_t j = 0; j < n; j += s2) {
            for (uint64_t k = 0; k < s1; ++k) {
                u = buf[j + k], v = buf[j + k + s1];
                buf[j + k] = u + v;
                buf[j + k + s1] = u - v;
            }
        }
        s1 <<= 1; s2 <<= 1;
    }
}


template<typename T>
constexpr bool has_vneg(const T& vec) {for(const auto &el: vec){ if(el < 0) return true;} return false;}


}

#endif // #ifndef _GFRP_FSM_H__
