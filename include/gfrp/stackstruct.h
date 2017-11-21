#ifndef _GFRP_STACKSTRUCT_H__
#define _GFRP_STACKSTRUCT_H__
#include <fstream>
#include "gfrp/util.h"
#include "FFHT/fht.h"
#include "fftw3.h"

namespace gfrp {

namespace fft {


template<typename FloatType>
struct FFTTypes;

template<>
struct FFTTypes<float> {
    using PlanType = fftwf_plan;
    using ComplexType = fftwf_complex;
    using FloatType = float;
    static constexpr decltype(&fftwf_destroy_plan) destroy_fn = &fftwf_destroy_plan;
    static constexpr decltype(&fftwf_execute_r2r)     r2rexec = &fftwf_execute_r2r;
    static constexpr decltype(&fftwf_execute_dft_r2c) r2cexec = &fftwf_execute_dft_r2c;
    static constexpr decltype(&fftwf_execute_dft_c2r) c2rexec = &fftwf_execute_dft_c2r;
    static constexpr decltype(&fftwf_execute_dft)     c2cexec = &fftwf_execute_dft;
    static constexpr decltype(&fftwf_sprint_plan) sprintfn = &fftwf_sprint_plan;
    static constexpr decltype(&fftwf_fprint_plan) fprintfn = &fftwf_fprint_plan;
    static constexpr decltype(&fftwf_cleanup)     cleanupfn= &fftwf_cleanup;
    static constexpr decltype(&fftwf_flops)       flopsfn = &fftwf_flops;
    static constexpr decltype(&fftwf_cost)        costfn = &fftwf_cost;
    static constexpr decltype(&fftwf_plan_dft_r2c) r2cplan = &fftwf_plan_dft_r2c;
    static constexpr decltype(&fftwf_plan_dft_c2r) c2rplan = &fftwf_plan_dft_c2r;
    static constexpr decltype(&fftwf_plan_dft) c2cplan = &fftwf_plan_dft;
    static constexpr decltype(&fftwf_plan_r2r) r2rplan = &fftwf_plan_r2r;
    static constexpr decltype(&fftwf_plan_r2r_1d) r2rplan1d = &fftwf_plan_r2r_1d;
    static constexpr decltype(&fftwf_import_wisdom_from_filename) loadfn = &fftwf_import_wisdom_from_filename;
    static constexpr decltype(&fftwf_export_wisdom_to_filename) storefn = &fftwf_export_wisdom_to_filename;
    static const char *suffix() {return "f";}
};
template<>
struct FFTTypes<double> {
    using PlanType = fftw_plan;
    using ComplexType = fftw_complex;
    using FloatType = double;
    static constexpr decltype(&fftw_destroy_plan) destroy_fn = &fftw_destroy_plan;
    static constexpr decltype(&fftw_execute_r2r)     r2rexec = &fftw_execute_r2r;
    static constexpr decltype(&fftw_execute_dft_r2c) r2cexec = &fftw_execute_dft_r2c;
    static constexpr decltype(&fftw_execute_dft_c2r) c2rexec = &fftw_execute_dft_c2r;
    static constexpr decltype(&fftw_execute_dft)     c2cexec = &fftw_execute_dft;
    static constexpr decltype(&fftw_sprint_plan) sprintfn = &fftw_sprint_plan;
    static constexpr decltype(&fftw_fprint_plan) fprintfn = &fftw_fprint_plan;
    static constexpr decltype(&fftw_cleanup)     cleanupfn= &fftw_cleanup;
    static constexpr decltype(&fftw_flops)       flopsfn = &fftw_flops;
    static constexpr decltype(&fftw_cost)        costfn = &fftw_cost;
    static constexpr decltype(&fftw_plan_dft_r2c) r2cplan = &fftw_plan_dft_r2c;
    static constexpr decltype(&fftw_plan_dft_c2r) c2rplan = &fftw_plan_dft_c2r;
    static constexpr decltype(&fftw_plan_dft) c2cplan = &fftw_plan_dft;
    static constexpr decltype(&fftw_plan_r2r) r2rplan = &fftw_plan_r2r;
    static constexpr decltype(&fftw_plan_r2r_1d) r2rplan1d = &fftw_plan_r2r_1d;
    static constexpr decltype(&fftw_import_wisdom_from_filename) loadfn = &fftw_import_wisdom_from_filename;
    static constexpr decltype(&fftw_export_wisdom_to_filename) storefn = &fftw_export_wisdom_to_filename;
    static const char *suffix() {return "d";}
};
template<>
struct FFTTypes<long double> {
    using PlanType = fftwl_plan;
    using ComplexType = fftwl_complex;
    using FloatType = double;
    static constexpr decltype(&fftwl_destroy_plan) destroy_fn = &fftwl_destroy_plan;
    static constexpr decltype(&fftwl_execute_r2r)     r2rexec = &fftwl_execute_r2r;
    static constexpr decltype(&fftwl_execute_dft_r2c) r2cexec = &fftwl_execute_dft_r2c;
    static constexpr decltype(&fftwl_execute_dft_c2r) c2rexec = &fftwl_execute_dft_c2r;
    static constexpr decltype(&fftwl_execute_dft)     c2cexec = &fftwl_execute_dft;
    static constexpr decltype(&fftwl_sprint_plan) sprintfn = &fftwl_sprint_plan;
    static constexpr decltype(&fftwl_fprint_plan) fprintfn = &fftwl_fprint_plan;
    static constexpr decltype(&fftwl_cleanup)     cleanupfn= &fftwl_cleanup;
    static constexpr decltype(&fftwl_flops)       flopsfn = &fftwl_flops;
    static constexpr decltype(&fftwl_cost)        costfn = &fftwl_cost;
    static constexpr decltype(&fftwl_plan_dft_r2c) r2cplan = &fftwl_plan_dft_r2c;
    static constexpr decltype(&fftwl_plan_dft_c2r) c2rplan = &fftwl_plan_dft_c2r;
    static constexpr decltype(&fftwl_plan_dft) c2cplan = &fftwl_plan_dft;
    static constexpr decltype(&fftwl_plan_r2r) r2rplan = &fftwl_plan_r2r;
    static constexpr decltype(&fftwl_plan_r2r_1d) r2rplan1d = &fftwl_plan_r2r_1d;
    static constexpr decltype(&fftwl_import_wisdom_from_filename) loadfn = &fftwl_import_wisdom_from_filename;
    static constexpr decltype(&fftwl_export_wisdom_to_filename) storefn = &fftwl_export_wisdom_to_filename;
    static const char *suffix() {return "ld";}
};

} // namespace fft


template<typename VecType>
void fht(VecType &vec) {
    if(vec.size() & (vec.size() - 1)) {
        throw runtime_error(ks::sprintf("vec size %zu not a power of two. NotImplemented.", vec.size()).data());
    } else {
        ::fht(&vec[0], log2_64(vec.size()));
    }
    vec *= 1. / std::sqrt(vec.size());
}

template<template<typename, bool> typename VecType, typename FloatType, bool VectorOrientation, typename=enable_if_t<is_floating_point<FloatType>::value>>
void fht(VecType<FloatType, VectorOrientation> &vec) {
    if(vec.size() & (vec.size() - 1)) {
        throw runtime_error(ks::sprintf("vec size %zu not a power of two. NotImplemented.", vec.size()).data());
    } else {
        ::fht(&vec[0], log2_64(vec.size()));
    }
    vec *= 1. / std::sqrt(vec.size());
}

template<typename Container>
struct is_dense_single {
    static constexpr bool value = blaze::IsDenseVector<Container>::value || blaze::IsDenseMatrix<Container>::value;
};

template<typename C1, typename C2>
struct is_dense {
    static constexpr bool value = is_dense_single<C1>::value && is_dense_single<C2>::value;
};


template<typename VecType1, typename VecType2>
void fht(const VecType1 &in, VecType2 &out) {
    std::fprintf(stderr, "About to call fht on sizes of %zu in and %zu out.\n", in.size(), out.size());
    static_assert(is_same<typename VecType1::ElementType, typename VecType2::ElementType>::value, "Input vectors must have the same type.");
    if constexpr(is_dense<VecType1, VecType2>::value) {
        fast_copy(&out[0], &in[0], sizeof(in[0]) * out.size());
        fht(out);
        return;
    } else {
        if constexpr(blaze::TransposeFlag<VecType1>::value == blaze::TransposeFlag<VecType2>::value) {
            if(out.size() == in.size()) {
                out = in;
                fht(out);
            } else throw runtime_error("NotImplemented.");
        } else {
            if(out.size() == in.size()) {
                out = transpose(in);
                fht(out);
            } else throw runtime_error("NotImplemented.");
        }
    }
    std::fprintf(stderr, "Called fht on sizes of %zu in and %zu out.\n", in.size(), out.size());
}

struct HadamardBlock {
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        if(out.size() == in.size()) {
            apply(out);
        } else {
            throw std::runtime_error("NotImplementedError: HadamardBlock for differing input/output sizes.");
        }
    }
    template<typename OutVector>
    void apply(OutVector &out) const {
        std::fprintf(stderr, "[%s] Calling fht on size %zu.\n", __PRETTY_FUNCTION__, out.size());
        if constexpr(blaze::IsSparseVector<OutVector>::value || blaze::IsSparseMatrix<OutVector>::value) {
            throw runtime_error("Fast Hadamard transform not implemented for sparse vectors yet.");
        }
        if((out.size() & (out.size() - 1)) == 0) {
            fht(out);
        } else {
            throw runtime_error("NotImplemented: either copy to another array, perform, and then subsample the last n rows, resize the output array.");
        }
        std::fprintf(stderr, "[%s] Called fht on size %zu.\n", __PRETTY_FUNCTION__, out.size());
    }
    template<typename FloatType>
    void apply(FloatType *pos, size_t nelem) const {
        std::fprintf(stderr, "[%s] Calling fht on size %zu.\n", __PRETTY_FUNCTION__, (size_t)(1ull << nelem));
        if(nelem > 48) {
            std::fprintf(stderr, "Warning: apply *should* take a log2 value. You're passing an impossibly large size.\n");
            nelem = log2_64(nelem);
        }
        ::fht(pos, nelem);
        const FloatType div(1./std::sqrt(static_cast<FloatType>(1 << nelem)));
        for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] *= div; // Could be vectorized.
        std::fprintf(stderr, "[%s] Called fht on size %zu.\n", __PRETTY_FUNCTION__, (size_t)(1ull << nelem));
    }
    template<typename IntType>
    void resize([[maybe_unused]] IntType i) {/* Do nothing */}
    template<typename IntType>
    void seed([[maybe_unused]] IntType i) {/* Do nothing */}
    HadamardBlock(size_t size=0) {
        if(size != 0) std::fprintf(stderr, "HadamardBlock being created with a nonzero size %zu\n", size);
    }
    size_t size() const {return -1;} // This is a lie.
};

namespace rfft {

static const char * names [] {
    "DCTBlock",
    "IDCTBlock",
};
static const fftw_r2r_kind kinds [] {
    FFTW_REDFT10,
    FFTW_REDFT01,
};

} // namespace rfft

template<typename FloatType>
class RFFTBlock {
    static_assert(is_floating_point<FloatType>::value, "RFFTBlock must be floating point.");
    using fftplan_t = typename fft::FFTTypes<FloatType>::PlanType;

    fftplan_t plan_;
    fftw_r2r_kind  kind_;
    int  n_, flags_;
    const bool oop_;


public:
    const char *block_type() const {
        auto it(std::find(std::begin(rfft::kinds), std::end(rfft::kinds), kind_));
        return it == std::end(rfft::kinds) ? "UNKNOWN" : rfft::names[it - std::begin(rfft::kinds)];
    }

    std::string wisdom_fname() const {
        return std::string(block_type()) + fft::FFTTypes<FloatType>::suffix();
    }
    void load_wisdom() {
        if(std::ifstream(wisdom_fname().data()).good()) {
            if(fft::FFTTypes<FloatType>::loadfn(wisdom_fname().data()) == 0) {
                throw std::runtime_error(ks::sprintf("Could not load wisdom from %s\n", wisdom_fname()).data());
            } else {
                std::fprintf(stderr, "Loaded wisdom from %s\n", wisdom_fname().data());
            }
        }
    }
    // Real FFT block
    void destroy() {
        if(plan_) {
            fft::FFTTypes<FloatType>::destroy_fn(plan_);
        }
    }
    void set_kind(fftw_r2r_kind kind) { kind_  = kind;}
    void set_flags(int newflags) { flags_ = newflags;}
    void resize(int n) {
        if(n == n_) return;
        n_ = n;
        blaze::DynamicVector<FloatType> tmpvec(n);
        auto ptr1(&tmpvec[0]);
        auto ptr2(oop_ ? ptr1: &blaze::DynamicVector<FloatType>(n)[0]);
        destroy();
        plan_ = fft::FFTTypes<FloatType>::r2rplan1d(n_, ptr1, ptr2, kind_, flags_);
    }
    RFFTBlock(int n, fftw_r2r_kind kind=FFTW_REDFT10,
              bool oop=false, int flags=FFTW_PATIENT):
                  plan_(nullptr), kind_(kind), n_(0), flags_(flags), oop_(oop) 
    {
        load_wisdom();
        resize(n);
    }
    template<typename VecType>
    void execute(VecType &a) {
        if(n_ != (int)a.size()) {
            resize((int)a.size());
        }
        fft::FFTTypes<FloatType>::r2rexec(plan_, &a[0], &a[0]);
        a *= std::sqrt(1./(a.size()<<1));
    }
    template<typename VecType1, typename VecType2>
    void execute(const VecType1 &in, VecType2 &out) {
        if(out.size() < in.size()) throw "ZOMG";
        if(n_ != in.size()) {
            resize((int)in.size());
        }
        fft::FFTTypes<FloatType>::r2rexec(plan_, &in[0], &out[0]);
        out *= std::sqrt(1./(out.size()<<1));
    }
    void execute(FloatType *a, FloatType *b) {
        if(plan_ == nullptr) throw runtime_error("ZOMG");
        fft::FFTTypes<FloatType>::r2rexec(plan_, a, b);
    }
    void store_wisdom() {
        if(fft::FFTTypes<FloatType>::storefn(wisdom_fname().data()) == 0) {
            std::fprintf(stderr, "Could not store wisdom at %s\n", wisdom_fname().data());
        }
    }
    ~RFFTBlock() {
        store_wisdom();
        destroy();
    }

    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        execute(in, out);
    }

    template<typename OutVector>
    void apply(OutVector &out) {
        execute(out);
    }

};

template<typename FloatType>
class DCTBlock: public RFFTBlock<FloatType> {
public:
    DCTBlock(int n, bool oop=false, int flags=FFTW_PATIENT): RFFTBlock<FloatType>(n, FFTW_REDFT10, oop, flags) {}
};
template<typename FloatType>
class IDCTBlock: public RFFTBlock<FloatType> {
public:
    IDCTBlock(int n, bool oop=false, int flags=FFTW_PATIENT): RFFTBlock<FloatType>(n, FFTW_REDFT01, oop, flags) {}
};

} // namespace gfrp

#endif // #ifndef _GFRP_STACKSTRUCT_H__
