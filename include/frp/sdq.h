#ifndef SORTED_DQ_H__
#define SORTED_DQ_H__
#include <cassert>
#include <deque>
#include <list>
#include <vector>
#include <algorithm>


namespace sorted {

// Sorted deque
template<template<typename...> class Container, typename T, typename All, typename Cmp=std::less<>, typename...Args>
class container {
    Container<T, All, Args...> data_;
    Cmp cmp_;
public:
    template<typename...CArgs>
    container(CArgs &&...args): data_(std::forward<CArgs>(args)...) {
        sort(data_.begin(), data_.end(), cmp_);
    }
    template<typename U>
    auto lower_bound(const U &item) const {
        return std::lower_bound(data_.begin(), data_.end(), item, cmp_);
    }
    auto find(const T &x) const {
        return std::lower_bound(data_.begin(), data_.end(), x, cmp_);
    }
    auto &con() {return data_;}
    auto &con() const {return data_;}
    template<typename...EArgs>
    auto emplace(EArgs &&...args) {
        T x(std::forward<EArgs>(args)...);
        auto it = find(x);
        data_.insert(it, std::move(x));
        assert(std::is_sorted(data_.begin(), data_.end(), cmp_));
    }
    T &operator[](size_t i) {return data_[i];}
    const T &operator[](size_t i) const {return data_[i];}
    auto begin() {return data_.begin();}
    auto end()   {return data_.end();}
    auto begin() const {return data_.begin();}
    auto end()   const {return data_.end();}
    auto cbegin() {return data_.cbegin();}
    auto cend()   {return data_.cend();}
    auto size() const {return data_.size();}
    auto pop() {auto ret = std::move(data_.back()); data_.pop_back(); return ret;}
    using iterator = typename Container<T, All>::iterator;
    using const_iterator = typename Container<T, All>::const_iterator;
    using value_type = typename Container<T, All>::value_type;
    using pointer = typename Container<T, All>::pointer;
    using const_pointer = typename Container<T, All>::const_pointer;
    using reference = typename Container<T, All>::reference;
    using const_reference = typename Container<T, All>::const_reference;
};

template<typename T, typename Cmp=std::less<>, typename All=std::allocator<T>>
using vector = container<std::vector, T, All, Cmp>;
template<typename T, typename Cmp=std::less<>, typename All=std::allocator<T>>
using deque = container<std::deque, T, All, Cmp>;

} // frp
#endif
