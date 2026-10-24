#include "nb_utils.hpp"
#include "single_engine.hpp"
#include "state/state.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(hpmc_ext, m) {
  using namespace hpmc;

  nb::enum_<ParticleType>(m, "ParticleType")
      .value("Sphere", ParticleType::Sphere);

  nb::class_<SingleBoxEngine>(m, "SingleBoxEngine")
      .def(
          "__init__",
          [](SingleBoxEngine *self, ParticleType type, double box_length,
             uint64_t seed, nb::kwargs kwargs) {
            ParticleConfig cfg;

            if (type == ParticleType::Sphere) {
              double radius = get_param<double>(kwargs, "radius", true);
              if (radius <= 0.0)
                throw nb::value_error("Sphere radius must be > 0.0");
              cfg = SphereConfig{radius};
            }

            new (self) SingleBoxEngine(type, box_length, cfg, seed);
          },
          nb::arg("type"), nb::arg("box_length"), nb::arg("seed") = 1,
          nb::arg("kwargs"))
      .def("n", [](SingleBoxEngine &e) { return e.state().n(); })
      .def("count_overlaps",
           [](SingleBoxEngine &e) { return e.state().count_global_overlaps(); })
      .def("run_batch", &SingleBoxEngine::run_batch, nb::arg("sweeps"),
           nb::arg("max_disp"))
      .def(
          "set_coords",
          [](SingleBoxEngine &e, nb::ndarray<double, nb::ndim<1>> x,
             nb::ndarray<nb::numpy, double, nb::ndim<1>> y,
             nb::ndarray<nb::numpy, double, nb::ndim<1>> z) {
            size_t N = x.shape(0);

            if (y.shape(0) != N || z.shape(0) != N) {
              throw nb::value_error(
                  "The coordinate arrays should have the same length");
            }

            std::span<const double> x_span(x.data(), N);
            std::span<const double> y_span(y.data(), N);
            std::span<const double> z_span(z.data(), N);

            e.state().set_coords(x_span, y_span, z_span);
          },
          nb::arg("x"), nb::arg("y"), nb::arg("z"))
      .def(
          "get_coords",
          [](SingleBoxEngine &e) {
            State &s = e.state();
            size_t shape = s.n();
            return std::make_tuple(nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                                       const_cast<double *>(s.x().data()), 1,
                                       &shape, nb::handle()),
                                   nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                                       const_cast<double *>(s.y().data()), 1,
                                       &shape, nb::handle()),
                                   nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                                       const_cast<double *>(s.z().data()), 1,
                                       &shape, nb::handle()));
          },
          nb::rv_policy::reference_internal);
}
