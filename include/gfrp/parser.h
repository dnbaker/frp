#ifndef _GFRP_PARSER_H__
#define _GFRP_PARSER_H__
#include <zlib.h>
#include "gfrp/util.h"

namespace gfrp {


namespace io {
template<typename FPType>
struct IOTypes;

inline size_t fgzread(FILE *fp, void *buf, unsigned len) {
    return fread(buf, 1, len, fp);
}

template<>
struct IOTypes<FILE *> {
    static constexpr decltype(&fopen) open = &fopen;
    static constexpr decltype(&fclose) close = &fclose;
    static constexpr decltype(&fgzread) read = &fgzread;
    static constexpr decltype(&feof) eof = &feof;
    static constexpr decltype(&ferror) error = &ferror;
};

template<>
struct IOTypes<gzFile> {
    static constexpr decltype(&gzopen) open = &gzopen;
    static constexpr decltype(&gzclose) close = &gzclose;
    static constexpr decltype(&gzread) read = &gzread;
    static constexpr decltype(&gzeof) eof = &gzeof;
    static constexpr decltype(&gzerror) error = &gzerror;
};

} // namespace io

#define USE_FP(attr) static constexpr auto attr = io::IOTypes<FPType>::attr

template<typename FPType, typename SizeType=unsigned>
class ChunkedLineReader {
    FPType      fp_;
    size_t   bufsz_;
    size_t   cnksz_;
    ks::string buf_;
    char       sep_;
    char line_char_;
    SizeType offset_;
    USE_FP(open);
    USE_FP(close);
    USE_FP(read);
    USE_FP(eof);
    USE_FP(error);
public:
    ChunkedLineReader(const char *path, const char *mode,
                      size_t bufsz=1 << 20,
                      size_t chunk_size=1 << 16,
                      char sep=',', char line_char='\n'):
        fp_(open(path, mode)), bufsz_(bufsz), cnksz_(chunk_size),
        buf_(bufsz), sep_(sep), line_char_(line_char), offset_(0)
    {
        if(fp_ == nullptr) throw std::bad_alloc();
    }
    struct LineIterator {
        ChunkedLineReader &ref_;
        
    };
    // TODO:
    // Add an iterator struct which emits a line.
    // Emit line as a view.
    // Add code outside of this which converts a delimited line into an array of floats.
    auto read_chunk() {
        auto ret(read(fp_, buf_.data(), std::min(cnksz_, buf_.size() - offset_)));
        offset_ += ret;
        buf_[offset_] = '\0';
    }
    int fill_buf() {
        SizeType c;
        while(offset_ + cnksz_ <= buf_.size()) {
            if((c = read_chunk()) != cnksz_) break;
        }
        int e;
        auto eofc(eof(fp_)), eerr(error(fp_));
        return eof(fp_) ? EOF: (e = error(fp_)) ? e: 0;
    }
    auto fp() {return fp_;}
    auto buf() {return buf_;}
    void set_sep(char sep) {sep_ = sep;}
    ks::string release_str() {
        ks::string ret;
        std::swap(ret, buf_);
        return ret;
    }
};

} // namespace gfrp

#endif
