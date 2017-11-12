#include "gfrp/parser.h"
using namespace gfrp;

int main(int argc, char *argv[]) {
    LineReader ic(argc > 1 ? argv[1]: "z.txt");
    unsigned i(0);
    std::vector<unsigned> tmp;
    blaze::DynamicVector<FLOAT_TYPE> vec(100000);
#if 0
    omp_set_num_threads(8);
#endif
    for(auto &line: ic) {
        i += line[0];
        line.set(vec);
        //std::fprintf(stderr, "line #%i is %s", ++i, line.data());
    }
    return EXIT_SUCCESS;
}
