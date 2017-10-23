#ifndef _GFRP_SCALING_H__
#define _GFRP_SCALING_H__
#include "gfrp/linalg.h"

namespace gfrp { scale {

#if 0
struct GaussianScale
{
   double operator()( double a ) const
   {
      return std::sqrt( a );
   }

   template< typename T >
   T load( const T& a ) const
   {
      return _mm256_sqrt_pd( a.value );
   }

   template< typename T >
   static constexpr bool simdEnabled() {
#if defined(__AVX__)
      return true;
#else
      return false;
#endif
   }
};
#endif

template<FloatType>
FloatType gaussian_scale(FloatType r, FloatType d) {
    // Using formula from Doubly Stochastic
    return std::pow(2. * M_PI, (-d * 0.5)) * std::exp(-.5 * r * r);
}


}}

#endif // #ifndef _GFRP_SCALING_H__
