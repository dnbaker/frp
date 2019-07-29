#ifndef SORTED_DQ_H__
#define SORTED_DQ_H__
#include <cassert>
#include <deque>
#include <list>
#include <algorithm>


namespace frp {

// Sorted deque
template<template<typename, typename> class Container, typename T, typename All>
class sorted_container {
    Container<T, All> data_;
public:
    template<typename...Args>
    sorted_container(Args &&...args): data_(std::forward<Args>(args)...) {
        sort(data_.begin(), data_.end());
    }
    auto find(const T &x) const {
        return std::lower_bound(data_.begin(), data_.end(), x);
    }
    auto &con() {return data_;}
    auto &con() const {return data_;}
    template<typename...Args>
    auto emplace(Args &&...args) {
        T x(std::forward<Args>(args)...);
        auto it = find(x);
        data_.insert(it, std::move(x));
        assert(std::is_sorted(data_.begin(), data_.end()));
    }
    T &operator[](size_t i) {return data_[i];}
    const T &operator[](size_t i) const {return data_[i];}
    auto begin() {return data_.begin();}
    auto end()   {return data_.end();}
    auto begin() const {return data_.begin();}
    auto end()   const {return data_.end();}
    auto cbegin() {return data_.cbegin();}
    auto cend()   {return data_.cend();}
    using iterator = typename std::deque<T, All>::iterator;
    using const_iterator = typename std::deque<T, All>::const_iterator;
    using value_type = T;
};
template<typename T, typename All=std::allocator<T>>
struct sdeque: public sorted_container<std::deque, T, All> {};
template<typename T, typename All=std::allocator<T>>
struct svector: public sorted_container<std::vector, T, All> {};

} // frp
#endif
