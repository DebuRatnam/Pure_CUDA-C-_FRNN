#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include "frnn_engine.h"

namespace nb = nanobind;

NB_MODULE(frnn_cuda, m) {
    m.doc() = "N-dimensional fixed-radius / KNN CUDA search engine (torch-free)";
    nb::class_<FRNNEngine>(m, "FRNNEngine")
        .def(nb::init<int>(), nb::arg("max_points"))
        .def("search", &FRNNEngine::search,
             nb::arg("points"), nb::arg("K"), nb::arg("radius"),
             "CPU path: flat AoS host list in, (idxs, dists) flat lists out.")
        .def("search_gpu", &FRNNEngine::search_gpu,
             nb::arg("dev_ptr"), nb::arg("N"), nb::arg("dim"),
             nb::arg("K"), nb::arg("radius"),
             "GPU path: raw SoA device pointer in, (d_idxs_ptr, d_dists_ptr) out.")
        .def("get_results", &FRNNEngine::get_results,
             nb::arg("N"), nb::arg("K"),
             "Copy the last search_gpu() results device->host.");
}
