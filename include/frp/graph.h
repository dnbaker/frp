#ifndef FROOPY_GRAPH_H__
#define FROOPY_GRAPH_H__
#include "./util.h"
#include <deque>
#include <set>

namespace frp {
inline namespace graph {

struct Emplacer {
    template<template<typename...> class Container, typename Value, typename...CArgs>
    static auto emplace(Container<CArgs...> &c, Value &&v) {
        c.emplace(std::move(v));
    }
    template<typename Value, typename...CArgs>
    static auto emplace(std::vector<CArgs...> &c, Value &&v) {
        c.emplace_back(std::move(v));
    }
    template<typename Value, typename...CArgs>
    static auto emplace(std::deque<CArgs...> &c, Value &&v) {
        c.emplace_back(std::move(v));
    }
};

// Representation 1: all nodes implicit
template<typename IndexType=::std::uint32_t, bool is_directed=false,
         template<typename...> class EdgeContainer=std::set>
class SparseGraph {
public:
    using index_type = IndexType;
    using edge_type = std::pair<index_type, index_type>;
protected:
    index_type n_;
    EdgeContainer<edge_type> edges_;
public:
    SparseGraph(index_type n=0): n_(n) {
        
    }
    void resize(index_type newn) {
        if(newn < n_)
            for(const auto &pair: edges_)
                if(pair.first > n_ || pair.second > n_)
                    throw std::runtime_error("Resizing leaves dangling edges.");
        n_ = newn;
    }
    void add(index_type lhs, index_type rhs) {
        add(std::make_pair(lhs, rhs));
    }
    void add(edge_type edge) {
        CONST_IF(!is_directed) {
            if(edge.first > edge.second)
                std::swap(edge.first, edge.second);
        }
        if(std::max(edge.first, edge.second) > n_)
            throw std::runtime_error("Can't add edges between nodes that don't exist");
        Emplacer::emplace(edges_, edge);
    }
    void sort() {
        std::sort(edges_.begin(), edges_.end());
    }
};
// Representation 2: nodes explicit, with values
//                   edges are unweighted
template<typename ValueType,
         typename IndexType=::std::uint32_t, bool is_directed=false,
         template<typename...> class EdgeContainer=std::set,
         template<typename...> class NodeContainer=std::vector>
class NodeValuedSparseGraph: public SparseGraph<IndexType, is_directed, EdgeContainer> {
protected:
    using super = SparseGraph<IndexType, is_directed, EdgeContainer>;
    using node_type = ValueType;
    using super::edge_type;
    using super::index_type;
    NodeContainer<ValueType> nodes_;
public:
    template<typename...Args>
    NodeValuedSparseGraph(Args &&...args): nodes_(std::forward<Args>(args)...) {
        if(nodes_.size())
            super::resize(nodes_.size());
    }
    template<typename...Args>
    auto emplace_node(Args &&...args) {
        ++this->n_;
        return Emplacer::emplace(nodes_, std::forward<Args>(args)...);
    }
};

// TODO: Representation 3: nodes implicit, weighted edges
// TODO: Representation 4: nodes explicit, weighted edges
} // graph
} // frp

#endif /* FROOPY_GRAPH_H__ */
