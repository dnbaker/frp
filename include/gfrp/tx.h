#ifndef _GFRP_TX_H__
#define _GFRP_TX_H__
#include "blaze/Math.h"
#include "gfrp/rand.h"

// Performs transformations on matrices and vectors.

namespace gfrp {

template<class Container>
void fill_shuffled(Container &con) {
    std::iota(std::begin(con), std::end(con), static_cast<std::decay_t<decltype(con[0])>>(0));
#if USE_STD
    std::random_shuffle(std::begin(con), std::end(con));
#else
    for(size_t i(con.size()); i > 1; std::swap(con[rng::random_bounded_nearlydivisionless64(i-1)], con[i-1]), --i);
#endif
}

template<class Container, typename... Args>
Container make_shuffled(Args &&...args) {
    Container con(std::forward<Args>(args)...);
    fill_shuffled(con);
    return con;
}

} //namespace gfrp::tx

#endif  // _GFRP_TX_H__
