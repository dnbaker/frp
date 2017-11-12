#include "gfrp/gfrp.h"
#include <fstream>
#include <getopt.h>
using namespace gfrp;

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    int co, nd(-1), target_dim(-1);
    size_t seed(-1);
    while((co = getopt(argc, argv, "s:m:n:h?")) >= 0) {
        switch(co) {
            case 'n': nd = std::atoi(optarg); break;
            case 'm': target_dim = std::atoi(optarg); break;
            case 's': seed = strtoull(optarg, nullptr, 10); break;
            case 'h': case '?': usage: {
                std::fprintf(stderr, "%s <args> input.path <output.path [defaults to stdout]\n"
                                     "-n:\tNumber of dimensions of input data.\n"
                                     "-m:\tNumber of dimensions to project to. (ust be <= n)\n",
                             argv[0]);
                std::exit(1);
            }
        }
    }
    size_t vecsize(roundup64(nd));
    
    if(nd < 0 || target_dim < 0 || target_dim >= nd) {
        goto usage;
    }
    LineReader ic(argv[optind]);
    std::ofstream ofs(optind + 1 < argc ? argv[optind + 1]: "/dev/stdout");
    COJLT<FLOAT_TYPE, 3> jl(nd, target_dim, seed);
    std::vector<unsigned> tmp;
    blaze::DynamicVector<FLOAT_TYPE> vec(vecsize);
#if USE_OPENMP
    omp_set_num_threads(8);
#endif
    for(auto &line: ic) {
        line.set(vec, tmp);
        jl.transform_inplace(vec);
        ofs << subvector(vec, 0, target_dim); // Maybe change output later.
        //std::fprintf(stderr, "line #%i is %s", ++i, line.data());
    }
    return EXIT_SUCCESS;
}
