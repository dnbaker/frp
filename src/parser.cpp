#include "gfrp/parser.h"

int main(int argc, char *argv[]) {
    IOChunker<FILE *> ic(argc > 1 ? argv[1]: "/dev/urandom", "rb");
    
}
