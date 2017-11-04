#ifndef _GFRP_STACKSTRUCT_H__
#define _GFRP_STACKSTRUCT_H__
#include "gfrp/util.h"
#include "FFHT/fht.h"
#include "fftw3.h"
#include "fftwrapper/fftwrapper.h"

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
    static constexpr decltype(&fftwl_plan_dft_c2r) c2cplan = &fftwl_plan_dft_c2r;
};

}


template<template<typename, bool> typename VecType, typename FloatType, bool VectorOrientation, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
void fht(VecType<FloatType, VectorOrientation> &vec) {
    if(vec.size() & (vec.size() - 1)) {
        throw std::runtime_error(ks::sprintf("vec size %zu not a power of two. NotImplemented.", vec.size()).data());
    } else {
        ::fht(&vec[0], log2_64(vec.size()));
    }
    vec *= 1. / std::sqrt(vec.size());
}

template<typename Container>
struct is_dense_single {
    static constexpr bool value = blaze::IsDenseVector<Container>::value || blaze::IsDenseMatrix<Container>::value;
};

#if 0
template<typename... Containers>
struct is_dense {
    static constexpr bool value = is_dense_single<Containers>::value && ...;
};
#endif
template<typename C1, typename C2>
struct is_dense {
    static constexpr bool value = is_dense_single<C1>::value && is_dense_single<C2>::value;
};


template<typename VecType1, typename VecType2>
void fht(const VecType1 &in, VecType2 &out) {
    static_assert(std::is_same<typename VecType1::ElementType, typename VecType2::ElementType>::value, "Input vectors must have the same type.");
    if constexpr(is_dense<VecType1, VecType2>::value) {
        fast_copy(&out[0], &in[0], sizeof(in[0]) * out.size());
        fht(&out[0], log2_64(out.size()));
        return;
    } else {
        if constexpr(blaze::TransposeFlag<VecType1>::value == blaze::TransposeFlag<VecType2>::value) {
            if(out.size() == in.size()) {
                out = in;
                fht(out);
            } else throw std::runtime_error("NotImplemented.");
        } else {
            if(out.size() == in.size()) {
                out = transpose(in);
                fht(out);
            } else throw std::runtime_error("NotImplemented.");
        }
    }
}

template<typename FloatType, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
struct HadamardBlock {
    size_t n_;
    size_t pow2up_;
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        throw std::runtime_error("NotImplemented.");
    }

    template<typename OutVector>
    void apply(OutVector &out) {
        if constexpr(blaze::IsSparseVector<OutVector>::value || blaze::IsSparseMatrix<OutVector>::value) {
            throw std::runtime_error("Fast Hadamard transform not implemented for sparse vectors yet.");
        }
        if(out.size() & (out.size() - 1) == 0) {
            fht(&out[0], log2_64(out.size()));
            return;
        }
        throw std::runtime_error("NotImplemented.");
    }
    HadamardBlock(size_t n): n_(n), pow2up_(roundup64(n_)) {}
};



template<typename FloatType, fftw_r2r_kind R2R_KIND, int FLAGS=FFTW_PATIENT>
struct RFFTBlock {
    static_assert(std::is_floating_point<FloatType>::value, "RFFTBlock must be floating point.");
    using fftplan_t = typename fft::FFTTypes<FloatType>::PlanType;
    static constexpr const char *WISDOM_PATH = "wisdom/RFFTBlock.wisdom";

    fftplan_t plan_;
    int n_;
    const bool oop_;
    // Real FFT block
    void destroy() {
        if(plan_) {
            fft::FFTTypes<FloatType>::destroy_fn(plan_);
        }
    }
    std::string fname_wisdom(const char *prefix) const {return std::string(prefix) + WISDOM_PATH;}
    void resize(int n) {
        n = n_;
        blaze::DynamicVector<FloatType> tmpvec(n);
        auto ptr1(&tmpvec[0]);
        auto ptr2(oop_ ? ptr1: &blaze::DynamicVector<FloatType>(n)[0]);
        destroy();
        plan_ = fft::FFTTypes<FloatType>::r2rplan1d(n_, ptr1, ptr2, R2R_KIND, FLAGS);
    }
    RFFTBlock(int n, bool oop=false): plan_(nullptr), n_(n), oop_(oop) {
        resize(n);
        if(std::experimental::filesystem::exists(WISDOM_PATH)) {
            fftw_import_wisdom_from_filename(WISDOM_PATH);
        }
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
        if(plan_ == nullptr) throw std::runtime_error("ZOMG");
        fft::FFTTypes<FloatType>::r2rexec(plan_, a, b);
    }
    ~RFFTBlock() {
        if(fftw_export_wisdom_to_filename(WISDOM_PATH) == 0) {
            std::cerr << "Warning: could not export wisdom to " << WISDOM_PATH << '\n';
        }
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

template<typename FloatType, int FLAGS=FFTW_PATIENT>
struct DCTBlock: public RFFTBlock<FloatType, FFTW_REDFT10, FLAGS> {
    static constexpr const char *WISDOM_PATH = "wisdom/DCTBlock.wisdom";
    template<typename... Args>
    DCTBlock(Args &&... args): RFFTBlock<FloatType, FFTW_REDFT10, FLAGS>(std::forward<Args>(args)...) {}
};

} // namespace gfrp

#endif // #ifndef _GFRP_STACKSTRUCT_H__
