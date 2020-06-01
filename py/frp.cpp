#include "FFHT/fht_header_only.h"
#include "frp/jl.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
using namespace frp;

namespace py = pybind11;

PYBIND11_MODULE(frp, m) {
    m.doc() = "Python bindings for frp"; // optional module docstring
    py::class_<OJLTransform> (m, "ojlt")
        .def(py::init<size_t, size_t, uint64_t, size_t>(), "Initialize an orthogonal JL transform from `from` to `do` dimensions, using seed `seed` and `nblocks` blocks",
             py::arg("from"), py::arg("to"), py::arg("seed")=137, py::arg("nblocks")=3)
        .def("resize", &OJLTransform::resize, "Resize the JL transform. This always rounds up to the nearest power of two.")
        .def("apply", [](const OJLTransform &jlt, py::array_t<double> input) -> py::array_t<double> {
            throw std::runtime_error("ojlt operates on floats, not doubles. Create an ojlt_d object instead.");
            // TODO: Add 2-d arrays and add kwarg for axis=0/1.
            return input;
        }, "Apply JL transform on double array, copying the input to an output array before performing.")
        .def("apply", [](const OJLTransform &jlt, py::array_t<float> input) -> py::array_t<float> {
            auto buf = input.request();
            auto result = py::array_t<float>(roundup(buf.size));
            auto resbuf(result.request());
            //std::memset(resbuf.ptr, 0, resbuf.size * resbuf.itemsize);
            std::memcpy(resbuf.ptr, buf.ptr, buf.size * buf.itemsize);
            jlt.transform_inplace((float *)resbuf.ptr);
            return result;
        }, "Apply JL transform on float array, copying the input to an output array before performing.")
        .def("apply_inplace", [](const OJLTransform &jlt, py::array_t<double> input) -> py::array_t<double> {
            throw std::runtime_error("ojlt operates on floats, not doubles. Create an ojlt_d object instead.");
            return input;
        }, "Apply JL transform on double array in-place, resizing up to nearest power of two if necessary.")
        .def("apply_inplace", [](const OJLTransform &jlt, py::array_t<float> input) -> py::array_t<float> {
            auto buf = input.request();
            if(buf.size & (buf.size - 1)) throw std::runtime_error("In place transform must be a power of two.");
            jlt.transform_inplace((float *)buf.ptr);
            return input;
        }, "Apply JL transform on double array in-place, resizing up to nearest power of two if necessary.")
        .def("reseed", &OJLTransform::reseed)
        .def("from_size", &OJLTransform::from_size)
        .def("to_size", &OJLTransform::to_size)
        .def("matrix_apply_oop", [](const OJLTransform &jlt, py::array_t<double> input) -> py::array_t<double> {
            throw std::runtime_error("ojlt operates on floats, not doubles. Create an ojlt_d object instead.");
            return input;
        }, "Apply a JL transform across a full vector, returning a result out of place.")
        .def("matrix_apply_oop", [](const OJLTransform &jlt, py::array_t<float> input) -> py::array_t<float> {
            py::buffer_info info = input.request();
            if(info.ndim != 2) throw std::runtime_error("OJL can only be called on matrices of 2 dimensions.");
            //std::fprintf(stderr, "Input array is %zu elements long each of dim %zu\n", info.shape[0], info.shape[1]);
            const ssize_t dest_size(jlt.to_size());
            py::array_t<float> ret(py::array_t<float>::ShapeContainer({info.shape[0], dest_size}));
            const size_t scratch_size(roundup(info.shape[1]));
            blaze::DynamicVector<float> tmp(scratch_size);
            py::buffer_info retinfo = ret.request();
            float *data((float *)info.ptr), *tmpdata(&tmp[0]);
            const size_t rowlen(info.shape[1]);
            if(rowlen & (rowlen - 1)) {
                //std::fprintf(stderr, "rowlen: %zu. Num rows: %zd\n", rowlen, info.shape[0]);
                auto tmpsub(subvector(tmp, 0, info.shape[1]));
                auto zsub(subvector(tmp, info.shape[1], scratch_size - info.shape[1]));
                for(ssize_t i(0); i < info.shape[0]; ++i) {
                    auto cv = blaze::CustomVector<float, blaze::unaligned, blaze::unpadded>(&data[rowlen * i], rowlen);
                    tmpsub = cv;
                    zsub = 0.f;
                    jlt.transform_inplace(tmp);
                    cv = tmpsub;
                }
            } else {
                for(ssize_t i(0); i < info.shape[0]; ++i) {
                    jlt.transform_inplace(&data[i * info.shape[1]]);
                }
            }
            return ret;
        }, "Apply a JL transform across a full vector, returning a result out of place.")
        .doc() = "Orthogonal JL transform for float32s";
    py::class_<DOJ> (m, "ojlt_d")
        .def(py::init<size_t, size_t, uint64_t, size_t>(), "Initialize an orthogonal JL transform from `from` to `do` dimensions, using seed `seed` and `nblocks` blocks",
             py::arg("from"), py::arg("to"), py::arg("seed")=137, py::arg("nblocks")=3)
        .def("resize", &DOJ::resize, "Resize the JL transform. This always rounds up to the nearest power of two.")
        .def("apply", [](const DOJ &jlt, py::array_t<double> input) -> py::array_t<double> {
            // TODO: Add 2-d arrays and add kwarg for axis=0/1.
            auto buf = input.request();
            auto result = py::array_t<double>(roundup(buf.size));
            auto resbuf(result.request());
            //std::memset(resbuf.ptr, 0, resbuf.size * resbuf.itemsize);
            std::memcpy(resbuf.ptr, buf.ptr, buf.size * buf.itemsize);
            jlt.transform_inplace((double *)resbuf.ptr);
            return result;
        }, "Apply JL transform on double array, copying the input to an output array before performing.")
        .def("apply", [](const DOJ &jlt, py::array_t<float> input) -> py::array_t<float> {
            throw std::runtime_error("ojlt_d operates on doubles, not floats. Create an ojlt object instead.");
            return input;
        }, "Apply JL transform on float array, copying the input to an output array before performing.")
        .def("apply_inplace", [](const DOJ &jlt, py::array_t<double> input) -> py::array_t<double> {
            auto buf = input.request();
            if(buf.size & (buf.size - 1)) throw std::runtime_error("In place transform must be a power of two.");
            jlt.transform_inplace((double *)buf.ptr);
            return input;
        }, "Apply JL transform on double array in-place, resizing up to nearest power of two if necessary.")
        .def("apply_inplace", [](const DOJ &jlt, py::array_t<float> input) -> py::array_t<float> {
            throw std::runtime_error("ojlt_d operates on doubles, not floats. Create an ojlt object instead.");
            return input;
        }, "Apply JL transform on double array in-place, resizing up to nearest power of two if necessary.")
        .def("reseed", &DOJ::reseed)
        .def("from_size", &DOJ::from_size)
        .def("to_size", &DOJ::to_size)
        .def("matrix_apply_oop", [](const DOJ &jlt, py::array_t<double> input) -> py::array_t<double> {
            py::buffer_info info = input.request();
            if(info.ndim != 2) throw std::runtime_error("OJL can only be called on matrices of 2 dimensions.");
            //std::fprintf(stderr, "Input array is %zu elements long each of dim %zu\n", info.shape[0], info.shape[1]);
            const ssize_t dest_size(jlt.to_size());
            py::array_t<double> ret(py::array_t<double>::ShapeContainer({info.shape[0], dest_size}));
            const size_t scratch_size(roundup(info.shape[1]));
            blaze::DynamicVector<double> tmp(scratch_size);
            py::buffer_info retinfo = ret.request();
            double *data((double *)info.ptr), *tmpdata(&tmp[0]);
            const size_t rowlen(info.shape[1]);
            if(rowlen & (rowlen - 1)) {
                //std::fprintf(stderr, "rowlen: %zu. Num rows: %zd\n", rowlen, info.shape[0]);
                auto tmpsub(subvector(tmp, 0, info.shape[1]));
                auto zsub(subvector(tmp, info.shape[1], scratch_size - info.shape[1]));
                for(ssize_t i(0); i < info.shape[0]; ++i) {
                    auto cv = blaze::CustomVector<double, blaze::unaligned, blaze::unpadded>(&data[rowlen * i], rowlen);
                    tmpsub = cv;
                    zsub = 0.f;
                    jlt.transform_inplace(tmp);
                    cv = tmpsub;
                }
            } else {
                for(ssize_t i(0); i < info.shape[0]; ++i) {
                    jlt.transform_inplace(&data[i * info.shape[1]]);
                }
            }
            return ret;
        }, "Apply a JL transform across a full vector, returning a result out of place.")
        .def("matrix_apply_oop", [](const DOJ &jlt, py::array_t<float> input) -> py::array_t<float> {
            throw std::runtime_error("ojlt_d operates on doubles, not floats. Create an ojlt object instead.");
            return input;
        }, "Apply a JL transform across a full vector, returning a result out of place.")
        .doc() = "Orthogonal JL transform for doubles";
}
