#include "lsmc_engine.hpp"
#include "nanobind/ndarray.h"
#include "nb_utils.hpp"
#include "single_engine.hpp"
#include "state/state.hpp"

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(hpmc_ext, m) {
  using namespace hpmc;

  nb::enum_<ParticleType>(m, "ParticleType")
      .value("Sphere", ParticleType::Sphere);

  nb::class_<State>(m, "State")
      .def(
          "__init__",
          [](State *self, ParticleType type, double box_length,
             nb::kwargs kwargs) {
            ParticleConfig cfg;
            if (type == ParticleType::Sphere) {
              double radius = get_param<double>(kwargs, "radius", true);
              if (radius <= 0.0)
                throw nb::value_error("Sphere radius must be > 0.0");
              cfg = SphereConfig{radius};
            }
            new (self) State(type, box_length, cfg);
          },
          nb::arg("type"), nb::arg("box_length"), nb::arg("kwargs"))
      .def("n", &State::n)
      .def("count_overlaps", &State::count_global_overlaps)
      // Zero-copy array input using your span implementation
      .def(
          "set_coords",
          [](State &s, nb::ndarray<double, nb::ndim<1>, nb::c_contig> x,
             nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> y,
             nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> z) {
            size_t N = x.shape(0);
            if (y.shape(0) != N || z.shape(0) != N) {
              throw nb::value_error(
                  "The coordinate arrays must have the same length");
            }
            std::span<const double> x_span(x.data(), N);
            std::span<const double> y_span(y.data(), N);
            std::span<const double> z_span(z.data(), N);
            s.set_coords(x_span, y_span, z_span);
          },
          nb::arg("x"), nb::arg("y"), nb::arg("z"))

      // Zero-copy coordinate exposure via views
      .def(
          "get_coords",
          [](State &s) {
            size_t shape[1] = {s.n()};
            return std::make_tuple(
                nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>(
                    const_cast<double *>(s.x().data()), 1, shape, nb::handle()),
                nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>(
                    const_cast<double *>(s.y().data()), 1, shape, nb::handle()),
                nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>(
                    const_cast<double *>(s.z().data()), 1, shape,
                    nb::handle()));
          },
          nb::rv_policy::reference_internal);

  nb::class_<SingleBoxEngine>(m, "SingleBoxEngine")
      .def(
          "__init__",
          [](SingleBoxEngine *self, std::shared_ptr<State> state,
             uint64_t seed) { new (self) SingleBoxEngine(state, seed); },
          nb::arg("state"), nb::arg("seed") = 1)
      .def("run_batch", &SingleBoxEngine::run_batch, nb::arg("sweeps"),
           nb::arg("max_disp"));

  nb::enum_<Lattice>(m, "Lattice")
      .value("A", Lattice::A)
      .value("B", Lattice::B);

  nb::class_<LSMCEngine>(m, "LSMCEngine")
      .def(
          "__init__",
          [](LSMCEngine *self, std::shared_ptr<State> state_A,
             std::shared_ptr<State> state_B, int min_M, int max_M,
             uint64_t seed) {
            new (self) LSMCEngine(state_A, state_B, min_M, max_M, seed);
          },
          nb::arg("state_A"), nb::arg("state_B"), nb::arg("min_M"),
          nb::arg("max_M"), nb::arg("seed") = 1)
      .def_prop_ro("M", &LSMCEngine::M)
      .def_prop_ro("active_lattice", &LSMCEngine::active_lattice)
      .def_prop_rw("wl_ln_f", &LSMCEngine::get_wl_ln_f,
                   &LSMCEngine::set_wl_ln_f)
      .def("run_batch", &LSMCEngine::run_batch, nb::arg("sweeps"),
           nb::arg("max_disp"))
      // Expose the bias weights array to Python as a NumPy view
      .def_prop_ro(
          "bias_weights",
          [](LSMCEngine &e) {
            size_t shape = {e.bias_weights().size()};
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                e.bias_weights().data(), 1, &shape, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def(
          "set_bias_weights",
          [](LSMCEngine &e,
             nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>
                 new_bias) {
            std::span<double> span(new_bias.data(), new_bias.size());
            e.set_bias_weights(span);
          },
          nb::arg("new_bias"))

      // Expose the visit counts array to Python as a NumPy view
      .def_prop_ro(
          "visit_counts",
          [](LSMCEngine &e) {
            size_t shape = {e.visit_counts().size()};
            return nb::ndarray<nb::numpy, int, nb::ndim<1>>(
                e.visit_counts().data(), 1, &shape, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def("reset_visit_counts", &LSMCEngine::reset_visit_counts);
  ;
}
