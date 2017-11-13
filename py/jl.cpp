#include "gfrp/jl.h"
#include <pybind11/pybind11.h>
using namespace gfrp;
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}
PYBIND11_MODULE(jl, m) {
    m.doc() = "pybind11-powered orthogonal JL transform"; // optional module docstring
    py::class_<OJLTransform<3>>(m, "ojlt")
        .def(py::init<size_t, size_t, size_t>())
        .def("resize", &OJLTransform<3>::resize);
}
