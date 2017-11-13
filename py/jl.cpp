#include "gfrp/jl.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
using namespace gfrp;
namespace py = pybind11;

PYBIND11_MODULE(jl, m) {
    m.doc() = "pybind11-powered orthogonal JL transform"; // optional module docstring
    py::class_<OJLTransform<3>> (m, "ojlt")
        .def(py::init<size_t, size_t, size_t>())
        .def("resize", &OJLTransform<3>::resize, "Resize the JL transform. This always rounds up to the nearest power of two.")
        .def("apply", [&](const OJLTransform<3> &jlt, py::array_t<double> input) -> py::array_t<double> {
            auto buf = input.request();
            auto result = py::array_t<double>(roundup(buf.size));
            auto resbuf(result.request());
            std::memset(resbuf.ptr, 0, resbuf.size * resbuf.itemsize);
            std::memcpy(resbuf.ptr, buf.ptr, buf.size * buf.itemsize);
            jlt.transform_inplace((double *)resbuf.ptr);
            return result;
        }, "Apply JL transform on double array, copying the input to an output array before performing.")
        .def("apply", [&](const OJLTransform<3> &jlt, py::array_t<float> input) -> py::array_t<float> {
            auto buf = input.request();
            auto result = py::array_t<float>(roundup(buf.size));
            auto resbuf(result.request());
            std::memset(resbuf.ptr, 0, resbuf.size * resbuf.itemsize);
            std::memcpy(resbuf.ptr, buf.ptr, buf.size * buf.itemsize);
            jlt.transform_inplace((float *)resbuf.ptr);
            return result;
        }, "Apply JL transform on float array, copying the input to an output array before performing.")
        .def("apply_inplace", [&](const OJLTransform<3> &jlt, py::array_t<double> &input) -> py::array_t<double> & {
            input.resize({roundup(input.size())});
            auto buf = input.request();
            jlt.transform_inplace((double *)buf.ptr);
            return input;
        }, "Apply JL transform on double array in-place, resizing up to nearest power of two if necessary.")
        .def("apply_inplace", [&](const OJLTransform<3> &jlt, py::array_t<float> &input) -> py::array_t<float> & {
            input.resize({roundup(input.size())});
            auto buf = input.request();
            jlt.transform_inplace((float *)buf.ptr);
            return input;
        }, "Apply JL transform on double array in-place, resizing up to nearest power of two if necessary.")
        .def("reseed", py::overload_cast<size_t>(&OJLTransform<3>::reseed));
}
