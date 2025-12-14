#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array_t<double> tensor_mac(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> c) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    auto buf_c = c.request();

    if (buf_a.ndim != buf_b.ndim || buf_a.ndim != buf_c.ndim) {
        throw std::runtime_error(
            "Input tensors must have the same number of dimensions");
    }
    for (int i = 0; i < buf_a.ndim; ++i) {
        if (buf_a.shape[i] != buf_b.shape[i] ||
            buf_a.shape[i] != buf_c.shape[i]) {
            throw std::runtime_error("Input shapes must match");
        }
    }

    py::array_t<double> result(buf_a.shape, buf_a.strides);
    auto buf_res = result.request();

    double* ptr_a = static_cast<double*>(buf_a.ptr);
    double* ptr_b = static_cast<double*>(buf_b.ptr);
    double* ptr_c = static_cast<double*>(buf_c.ptr);
    double* ptr_res = static_cast<double*>(buf_res.ptr);

    // total number of elements
    size_t total = 1;
    for (ssize_t i = 0; i < buf_a.ndim; ++i)
        total *= static_cast<size_t>(buf_a.shape[i]);

    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; ++i) {
        ptr_res[i] = ptr_a[i] * ptr_b[i] + ptr_c[i];
    }

    return result;
}

PYBIND11_MODULE(_tensor_ops, m) {
    m.doc() = "High-performance Tensor MAC (A * B + C)";
    m.def("tensor_mac", &tensor_mac,
          "element-wise A * B + C for numerical numpy arrays");
}