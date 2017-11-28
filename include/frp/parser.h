#ifndef _GFRP_PARSER_H__
#define _GFRP_PARSER_H__
#include <zlib.h>
#include "frp/util.h"

namespace frp {


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

static const std::string zlibsuf  = ".gz";
static const std::string bzip2suf = ".bz2";
static const std::string zstdsuf  = ".zst";
static const std::string zlibcmd  = "gzip -dc ";
static const std::string bzip2cmd = "bzip2 -dc ";
static const std::string zstdcmd  = "ztd -dc ";

bool ends_with(const std::string &pat, const std::string &ref) {
    return std::equal(std::rbegin(pat), std::rend(pat), std::rbegin(ref));
}

enum CType {
    UNKNOWN = -1,
    UNCOMPRESSED = 0, // FILE *
    ZLIB  = 1, // .gz
    ZSTD  = 2, // .zstd
    BZIP2 = 3  // .bz2
};

CType infer_ctype(const std::string &path) {
    if(ends_with(zlibsuf, path))  return ZLIB;
    if(ends_with(bzip2suf, path)) return BZIP2;
    if(ends_with(zstdsuf, path))  return ZSTD;
    return UNCOMPRESSED;
}

} // namespace io

#define USE_FP(attr) static constexpr auto attr = io::IOTypes<FPType>::attr

class LineReader {
    FILE         *fp_;
    std::string path_;
    io::CType  ctype_;
    char       delim_;
    size_t     bufsz_;
    ssize_t      len_;
    char       *data_;
    const std::string comment_lines_;

    /*
      Reads through a file line by line just once. Will add more functionality later.
     */
public:
    LineReader(const char *path,
               char delim='\n', size_t bufsz=0, io::CType ctype=io::UNKNOWN, std::string comment_lines="#"):
        fp_(nullptr), path_(path), ctype_(ctype >= 0 ? ctype: io::infer_ctype(path_)),
        delim_(delim), bufsz_(bufsz),
        len_(0), data_(bufsz_ ? (char *)std::malloc(bufsz_): nullptr),
        comment_lines_(std::move(comment_lines))
    {
    }
    ~LineReader() {
        if(fp_) fclose(fp_);
        std::free(data_);
    }
    class LineIterator {
        LineReader &ref_;
    public:
        LineIterator(LineReader &ref):
            ref_(ref) {}
        LineIterator &operator*() {
            return *this;
        }
        LineIterator &operator++() {
            ref_.len_ = getdelim(&ref_.data_, &ref_.bufsz_, ref_.delim_, ref_.fp_);
            if(good())
                if(std::find(ref_.comment_lines_.begin(), ref_.comment_lines_.end(), ref_.data_[0]) != ref_.comment_lines_.end())
                    return this->operator++();
            return *this;
        }
        using uivec_t = std::vector<unsigned>;

        ssize_t len() const {return ref_.len();}
        char *data() {return ref_.data();}
        const char *data() const {return ref_.data();}
        bool operator!=([[maybe_unused]] const LineIterator &other) const {return good();}
        bool operator< ([[maybe_unused]] const LineIterator &other) const {return good();}
        char &operator[](size_t index) {return data()[index];}
        const char &operator[](size_t index) const {return data()[index];}
        bool good() const {return ref_.len_ != -1;}
        // TODO: speed this up by avoiding making a vector of positions and just parse in the first pass.
        template<template <typename, bool> typename VectorType, typename FloatType, bool Orientation>
        void set(VectorType<FloatType, Orientation> &ret, uivec_t &offsets, const int delim=',') {
            if constexpr(blaze::IsSparseVector<VectorType<FloatType, Orientation>>::value)
                blaze::reset(ret);
            ks::split(data(), delim, len(), offsets);
            if(offsets.size() != ret.size()) {
                ret.resize(offsets.size());
                std::fprintf(stderr, "Warning: ret is now %zu in size.\n", ret.size());
                //throw std::runtime_error(ks::sprintf("Wrong sizes. Number of fields: %zu. Size of array: %zu\n", offsets.size(), ret.size()).data());
            }
            size_t i;
#ifdef USE_OPENMP
#pragma message("use openmp")
            #pragma omp parallel for schedule(dynamic, 8192)
#endif
            for(i = 0; i < std::min(ret.size(), offsets.size()); ++i) {
                ret[i] = std::atof(data() + offsets[i]);
            }
            if constexpr(!blaze::IsSparseVector<VectorType<FloatType, Orientation>>::value) {
                std::memset(&ret[i], 0, (ret.size() - i) * sizeof(FloatType)); // Zero the last elements in array.
            }
        }
        template<template <typename, bool> typename VectorType, typename FloatType, bool Orientation>
        void set(VectorType<FloatType, Orientation> &ret, const int delim=',') {
            if constexpr(blaze::IsSparseVector<VectorType<FloatType, Orientation>>::value)
                blaze::reset(ret);
            char *p(data()), *line_end(p + len());
            size_t i(0), e(ret.size());
            while(p < line_end) {
                ret[i++] = std::atof(p);
                if(((p = std::strchr(p, delim)) == nullptr) | (i == e)) break;
                ++p;
            }
            if constexpr(!blaze::IsSparseVector<VectorType<FloatType, Orientation>>::value) {
                std::memset(&ret[i], 0, (ret.size() - i) * sizeof(FloatType)); // Zero the last elements in array.
            }
        }
        template<template <typename, bool> typename VectorType, typename FloatType, bool Orientation>
        int sparse_set(VectorType<FloatType, Orientation> &ret, const int delim=' ') {
            char *p(data()), *line_end(p + len());
            const int label(std::atoi(p));
            blaze::reset(ret);
            if((p = std::strchr(p, ' ')) == nullptr) return label;
            ++p;
            while(p < line_end) {
                ret[std::atoi(p) - 1] = std::atof(std::strchr(p, ':') + 1);
                if((p = std::strchr(p, delim)) == nullptr) break;
                ++p;
            }
            return label;
        }
    };
    LineIterator begin() {
        using namespace io;
        if(fp_) {
            fclose(fp_);
            std::fprintf(stderr, "Closing!\n");
        }
        std::fprintf(stderr, "Opening!\n");
        switch(ctype_) {
            case UNCOMPRESSED: fp_ = fopen(path_.data(), "r"); break;
            case ZLIB:         fp_ = popen((zlibcmd  + path_).data(), "r"); break;
            case BZIP2:        fp_ = popen((bzip2cmd + path_).data(), "r"); break;
            case ZSTD:         fp_ = popen((zstdcmd  + path_).data(), "r"); break;
            default:           throw std::runtime_error("Unexpected ctype code: " + std::to_string((int)ctype_));
        }
        if(fp_ == nullptr) {
            throw std::runtime_error(ks::sprintf("Could not open file at %s", path_.data()).data());
        }
        LineIterator ret(*this);
        return ++ret;
    }
    LineIterator end() {
        return LineIterator(*this);
    }
    ssize_t len() const {return len_;}
    char *data() {return data_;}
    const char *data() const {return data_;}
};

} // namespace frp

#endif
