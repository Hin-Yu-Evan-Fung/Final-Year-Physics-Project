#include "box_engine.hpp"
#include "lsmc_engine.hpp"
#include "nanobind/ndarray.h"
#include "nb_utils.hpp"
#include "state/state.hpp"

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using Arr1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;

NB_MODULE(hpmc_ext, m) {
  using namespace hpmc;

  nb::class_<SphereConfig>(m, "SphereConfig", "Hard Sphere Config.")
      .def(nb::init<double>(), nb::arg("radius"));

  nb::class_<Lattice>(m, "Lattice", "Triclinic lattice.")
      .def_static("from_cubic", &Lattice::from_cubic, nb::arg("l"),
                  "Create a cubic lattice with side length l.")
      .def_static(
          "from_box_lengths", &Lattice::from_box_lengths, nb::arg("lx"),
          nb::arg("ly"), nb::arg("lz"),
          "Create an orthorhombic lattice with side lengths (lx, ly, lz).")
      .def_static(
          "from_upper_triangular",
          [](const std::array<double, 3> &a, const std::array<double, 3> &b,
             const std::array<double, 3> &c) {
            return Lattice::from_upper_triangular(
                arr_to_vec3(a), arr_to_vec3(b), arr_to_vec3(c));
          },
          nb::arg("a"), nb::arg("b"), nb::arg("c"),
          "Create a general triclinic lattice from upper-triangular box "
          "vectors.")
      .def_ro("volume", &Lattice::volume, "Total volume of the simulation box.")
      .def_prop_ro("a", [](const Lattice &lat) { return vec3_to_arr(lat.a); })
      .def_prop_ro("b", [](const Lattice &lat) { return vec3_to_arr(lat.b); })
      .def_prop_ro("c", [](const Lattice &lat) { return vec3_to_arr(lat.c); })
      .def_prop_ro(
          "perp_widths",
          [](const Lattice &lat) { return vec3_to_arr(lat.perp_widths()); },
          "Perpendicular widths of the unit cell along each axis.")
      .def(
          "to_cart",
          [](const Lattice &lat, const std::array<double, 3> &s) {
            return vec3_to_arr(lat.to_cart(arr_to_vec3(s)));
          },
          nb::arg("s"),
          "Convert fractional coordinates to Cartesian coordinates.")
      .def(
          "to_frac",
          [](const Lattice &lat, const std::array<double, 3> &r) {
            return vec3_to_arr(lat.to_frac(arr_to_vec3(r)));
          },
          nb::arg("r"),
          "Convert Cartesian coordinates to fractional coordinates.")
      .def(
          "wrap",
          [](const Lattice &lat, const std::array<double, 3> &r) {
            return vec3_to_arr(lat.wrap(arr_to_vec3(r)));
          },
          nb::arg("r"),
          "Wrap Cartesian coordinates into the primary periodic box.");

  nb::class_<State>(m, "State",
                    "A state containing a periodic bounding box and particles "
                    "of the same type and config.")
      .def(nb::init<Config, Lattice>(), nb::arg("config"), nb::arg("lattice"))
      .def("n", &State::n,
           "Returns the total number of particles currently in the system.")
      .def("has_any_overlaps", &State::has_any_overlaps,
           "Returns True if any pair of particles overlaps.")
      .def("count_all_overlaps", &State::count_all_overlaps,
           "Returns the total number of particle overlaps in the system.")
      .def(
          "set_coords",
          [](State &s, Arr1D x, Arr1D y, Arr1D z) {
            if (y.size() != x.size() || z.size() != x.size()) {
              throw nb::value_error(
                  "The coordinate arrays must have the same length");
            }
            // Fixed: z.data(), z.size() (was y twice!)
            s.set_coords({x.data(), x.size()}, {y.data(), y.size()},
                         {z.data(), z.size()});
          },
          nb::arg("x"), nb::arg("y"), nb::arg("z"),
          "Sets particle Cartesian coordinates from 1D NumPy arrays (x, y, z).")
      .def(
          "get_coords",
          [](const State &s) {
            return std::make_tuple(as_view<const double>(s.x()),
                                   as_view<const double>(s.y()),
                                   as_view<const double>(s.z()));
          },
          nb::rv_policy::reference_internal,
          "Returns read-only NumPy views (x, y, z) of particle coordinates.");

  nb::class_<BoxEngine>(m, "BoxEngine",
                        "Single Box Monte Carlo simulation for any supported "
                        "rigid particle type.")
      .def(nb::init<const std::shared_ptr<State> &, uint64_t>(),
           nb::arg("state"), nb::arg("seed") = 0)
      .def("run_batch", &BoxEngine::run_batch, nb::arg("sweeps"),
           nb::arg("max_disp"), nb::call_guard<nb::gil_scoped_release>(),
           "Perform a batch of Monte Carlo sweeps at a given maximum "
           "displacement.");

  nb::enum_<LatticeType>(m, "LatticeType", "Active lattice identifier in LSMC.")
      .value("A", LatticeType::A)
      .value("B", LatticeType::B);

  nb::class_<LSMCEngine>(
      m, "LSMCEngine",
      "Lattice Switching Monte Carlo (LSMC) simulation engine.")
      .def(nb::init<std::shared_ptr<State>, std::shared_ptr<State>, int, int,
                    uint64_t>(),
           nb::arg("state_A"), nb::arg("state_B"), nb::arg("min_M"),
           nb::arg("max_M"), nb::arg("seed") = 0)
      .def_prop_ro("M", &LSMCEngine::M,
                   "Returns the current M (Overlaps A "
                   "- Overlaps B).")
      .def_prop_ro("min_M", &LSMCEngine::min_M, "Returns the minimum M.")
      .def_prop_ro("max_M", &LSMCEngine::max_M, "Returns the maximum M.")
      .def_prop_ro("active_lattice", &LSMCEngine::active_lattice,
                   "Returns whether the active lattice is currently "
                   "LatticeType.A or LatticeType.B.")
      .def_prop_rw("wl_ln_f", &LSMCEngine::get_wl_ln_f,
                   &LSMCEngine::set_wl_ln_f,
                   "Wang-Landau bias modification factor ln(f).")
      .def("run_batch", &LSMCEngine::run_batch, nb::arg("sweeps"),
           nb::arg("max_disp"), nb::call_guard<nb::gil_scoped_release>(),
           "Performs particle translation sweeps, records bias weights, and "
           "updates the macrostate.\n\n"
           "Returns:\n"
           "    The number of successfully accepted moves.")
      .def_prop_ro(
          "bias_weights",
          [](const LSMCEngine &e) {
            return as_view<const double>(e.bias_weights());
          },
          nb::rv_policy::reference_internal,
          "Returns a read-only view of the bias weights.")
      .def(
          "set_bias_weights",
          [](LSMCEngine &e, Arr1D arr) {
            e.set_bias_weights({arr.data(), arr.size()});
          },
          nb::arg("new_bias"),
          "Sets the Wang-Landau bias weights from a 1D array.")
      .def_prop_ro(
          "visit_counts",
          [](const LSMCEngine &e) {
            return as_view<const uint64_t>(e.visit_counts());
          },
          nb::rv_policy::reference_internal,
          "Returns a read-only view of the macrostate visit counts.")
      .def("reset_visit_counts", &LSMCEngine::reset_visit_counts,
           "Resets all macrostate visit counts to zero.");
}
