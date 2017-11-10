#include "gfrp/parser.h"
using namespace gfrp;

int main(int argc, char *argv[]) {
    ChunkedLineReader<FILE *> ic(argc > 1 ? argv[1]: "/dev/urandom", "rb");
    ChunkedLineReader<gzFile> gzic(argc > 1 ? argv[1]: "/dev/urandom", "rb");
    
}
