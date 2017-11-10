#ifndef _GFRP_PARSER_H__
#define _GFRP_PARSER_H__
#include <zlib.h>
#include "gfrp/util.h"

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
};

template<>
struct IOTypes<gzFile> {
    static constexpr decltype(&gzopen) open = &gzopen;
    static constexpr decltype(&gzclose) close = &gzclose;
    static constexpr decltype(&gzread) read = &gzread;
};

#define USE_FP(attr) static constexpr auto attr = IOTypes<FPType>::attr

template<typename FPType, typename SizeType=unsigned>
class IOChunker {
    FPType      fp_;
    size_t   bufsz_;
    size_t   cnksz_;
    ks::string buf_;
    char       sep_;
    char line_char_;
    std::vector<SizeType> starts_;
    unsigned nlines_;
    SizeType offset_;
    USE_FP(open);
    USE_FP(close);
    USE_FP(read);
public:
    IOChunker(const char *path, const char *mode,
              size_t bufsz=1 << 18, size_t chunk_size=1<<16,
              char sep=',', char line_char='\n'):
        fp_(open(path, mode)), bufsz_(bufsz), cnksz_(chunk_size),
        buf_(bufsz), sep_(sep), line_char_(line_char), nlines_(0), offset_(0)
    {
        if(fp_ == nullptr) throw std::bad_alloc();
    }
    SizeType read_chunk(SizeType pos) {
        auto nread(read(fp_ + pos, buf_.data(), std::min(cnksz_, bufsz_ - buf_.l)));
        buf_.l += nread;
        return nread;
    }
    int fill_buf() {
        SizeType c;
        SizeType pos(starts_.size() ? starts_.back(): 0);
        while((c = read_chunk(pos)) == cnksz_) pos += c;
        return (buf_.size() < bufsz_) ? 0: EOF;
    }
    void line_split() {
        auto ntoks(ks::split(buf_.data(), line_char_, buf_.size(), starts_));
    }
    template<typename VecType>
    void parse_next_line(VecType &data) {
        // Dense
        if(offset_ + 1 >= starts_.size()) {
            std::memmove(buf_.data(), buf_.data() + starts_.back(), buf_.size() - starts_.back());
            // STuff.
        }
        char *p(buf_.data() + starts_[offset_++]);
        auto toks(ks::split<SizeType>(p, sep_));
        if constexpr(std::is_arithmetic<std::decay_t<decltype(data[0])>>::value) {
            for(size_t i(0); i < toks.size(); ++i) {
                data[i] = std::atof(p + toks[i]);
            }
        }
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

#endif
