#include "gfrp/parser.h"
using namespace gfrp;

int main(int argc, char *argv[]) {
    LineReader ic(argc > 1 ? argv[1]: "z.txt");
    unsigned i(0);
    for(auto &line: ic) {
        i += line[0];
        //std::fprintf(stderr, "line #%i is %s", ++i, line.data());
    }
    return EXIT_SUCCESS;
}
