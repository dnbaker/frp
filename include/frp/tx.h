#ifndef _GFRP_TX_H__
#define _GFRP_TX_H__
#include "blaze/Math.h"
#include "frp/rand.h"

// Performs transformations on matrices and vectors.

namespace frp {

template<class Container>
void fill_shuffled(uint64_t seed, Container &con) {
    // Note: this does not permit the setting of a seed. This should probably be changed.
    std::iota(std::begin(con), std::end(con), static_cast<std::decay_t<decltype(con[0])>>(0));
#if USE_STD
    std::random_shuffle(std::begin(con), std::end(con));
#else
    aes::AesCtr<uint64_t> gen(seed);
    for(size_t i(con.size()); i > 1; std::swap(con[rng::random_bounded_nearlydivisionless64(i-1, gen)], con[i-1]), --i);
#endif
}

template<class Container, typename... Args>
Container make_shuffled(uint64_t seed, Args &&...args) {
    Container con(std::forward<Args>(args)...);
    fill_shuffled(seed, con);
    return con;
}

} //namespace frp::tx

#endif  // _GFRP_TX_H__
