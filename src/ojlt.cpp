#include "gfrp/gfrp.h"
#include <fstream>
#include <getopt.h>
#include <ctime>
using namespace gfrp;

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    int co, nd(-1), target_dim(-1);
    size_t seed(-1), vecbufsz(1 << 18);
    while((co = getopt(argc, argv, "s:m:n:th?")) >= 0) {
        switch(co) {
            case 'n': nd = atoi(optarg); break;
            case 'm': target_dim = atoi(optarg); break;
            case 's': seed = strtoull(optarg, nullptr, 10); break;
            case 't': seed = time(nullptr); break;
            case 'b': vecbufsz = strtoull(optarg, nullptr, 10); break;
            case 'h': case '?': usage: {
                fprintf(stderr, "%s <args> input.path <output.path [defaults to stdout]\n"
                                "-n:\tNumber of dimensions of input data.\n"
                                "-m:\tNumber of dimensions to project to. (ust be <= n)\n"
                                "-s:\tSeed RNG with [argument as unsigned long long]\n"
                                "-t:\tSeed RNG with std::time(nullptr)\n"
                                "-h:\tEmit usage\n",
                             argv[0]);
                exit(1);
            }
        }
    }
    
    if(target_dim < 0) {
        goto usage;
    }
    LineReader ic(argv[optind]);
    if(nd < 0) {
        auto fline(ic.begin());
        nd = countchars(fline.data(), ',') + 1;
        std::fprintf(stderr, "Counted %i fields\n", nd);
    }
    size_t vecsize(nd);
    if(target_dim >= nd) {
        goto usage;
    }
    FILE *ofp(optind + 1 < argc ? fopen(argv[optind + 1], "w"): stdout);
    const int fn(fileno(ofp));
    OJLTransform<3> jl(nd, target_dim, seed);
    ks::string str(vecbufsz);
#if PARALLEL_PARSE
    std::vector<unsigned> tmp;
#endif
    blaze::DynamicVector<FLOAT_TYPE> vec(roundup(vecsize));
#if USE_OPENMP
    omp_set_num_threads(8);
#endif
    for(auto &line: ic) {
#if PARALLEL_PARSE
        line.set(vec, tmp);
#else
        line.set(vec);
#endif
        jl.transform_inplace(vec);
        ksprint(subvector(vec, 0, target_dim), str);
        str.putc_('\n');
        if(str.size() & (~((str.capacity()>>1) - 1))) {
            // if str.size >= str.capacity/2
            str.write(fn);
            str.clear();
        }
    }
    str.write(fn);
    str.clear();
    if(ofp != stdout) fclose(ofp);
    return EXIT_SUCCESS;
}
