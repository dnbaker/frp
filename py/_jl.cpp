#include "gfrp/jl.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
using namespace gfrp;

namespace py = pybind11;

PYBIND11_MODULE(_jl, m) {
    m.doc() = "pybind11-powered orthogonal JL transform"; // optional module docstring
    py::class_<OJLTransform> (m, "ojlt")
        .def(py::init<size_t, size_t, uint64_t, size_t>())
        .def("resize", &OJLTransform::resize, "Resize the JL transform. This always rounds up to the nearest power of two.")
        .def("apply", [](const OJLTransform &jlt, py::array_t<double> input) -> py::array_t<double> {
            // TODO: Add 2-d arrays and add kwarg for axis=0/1.
            auto buf = input.request();
            auto result = py::array_t<double>(roundup(buf.size));
            auto resbuf(result.request());
            //std::memset(resbuf.ptr, 0, resbuf.size * resbuf.itemsize);
            std::memcpy(resbuf.ptr, buf.ptr, buf.size * buf.itemsize);
            jlt.transform_inplace((double *)resbuf.ptr);
            return result;
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
            auto buf = input.request();
            if(buf.size & (buf.size - 1)) throw std::runtime_error("In place transform must be a power of two.");
            jlt.transform_inplace((double *)buf.ptr);
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
            //std::fprintf(stderr, "rowlen: %zu. Num rows: %zd\n", rowlen, info.shape[0]);
            for(ssize_t i(0); i < info.shape[0]; ++i) {
                tmp.reset();
                std::memcpy((void *)tmpdata, (void *)&data[rowlen * i], rowlen * sizeof(double));
                jlt.transform_inplace(tmpdata);
                std::memcpy((void *)(((double *)retinfo.ptr) + dest_size * i), (void *)tmpdata, dest_size * sizeof(double));
            }
            return ret;
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
            //std::fprintf(stderr, "rowlen: %zu. Num rows: %zd\n", rowlen, info.shape[0]);
            for(ssize_t i(0); i < info.shape[0]; ++i) {
                tmp.reset();
                std::memcpy((void *)tmpdata, (void *)&data[rowlen * i], rowlen * sizeof(float));
                jlt.transform_inplace(tmpdata);
                std::memcpy((void *)(((float *)retinfo.ptr) + dest_size * i), (void *)tmpdata, dest_size * sizeof(float));
            }
            return ret;
        }, "Apply a JL transform across a full vector, returning a result out of place.");
}
