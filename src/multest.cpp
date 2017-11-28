#include "frp/frp.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std::chrono;
using namespace frp;
using namespace blaze;

template<typename T>
void print_vec(T &vec) {
    std::cerr << std::scientific;
    std::cerr << "[";
    for(auto el: vec) std::cerr << el << ",";
    std::cerr << "]\n";
}

struct BaseClass {
    std::string msg;
    BaseClass(std::string in): msg(in) {}
    virtual void set_string(std::string append) {
        msg += append;
    }
    void print_string(const std::string &to_app) {
        set_string(to_app);
        std::cerr << "New string " << msg << '\n';
    }
};

struct DerivedClass: BaseClass {
    DerivedClass(std::string in): BaseClass(in) {}
    virtual void set_string(std::string append) {
        msg += append + "DERIVED CLASS YO ";
    }
};

int main(int argc, char *argv[]) {
#if 0
    const unsigned len(argc == 1 ? 1 << 16 : std::atoi(argv[1]));
    DynamicVector<FLOAT_TYPE> vec(len);
    DynamicVector<FLOAT_TYPE> ret(len);
    for(auto &el: vec) el = FLOAT_TYPE(std::rand()) / RAND_MAX;
    std::cerr << "Making vec\n";
    PRNVector<aes::AesCtr<uint64_t>,
              unit_normal<FLOAT_TYPE>> pv(len);
    std::cerr << "Made \n";
    auto it(vec.begin());
    unsigned i(0);
    for(auto el: pv) {
        std::cerr << "Accessing index " << i + 1;
         *it = el;
        std::cerr << "pv[" << i << "] is " << pv[i] << '\n';
        ++it;
        ++i;
    }
#else
    BaseClass b("BaseYo");
    DerivedClass d("DerivedYo");
    b.print_string("YAY");
    d.print_string("YAY");
#endif
}
