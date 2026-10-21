#include "core/state.hpp"
#include "core/vec3.hpp"
#include "nb_utils.hpp"
#include "particles/hard_sphere.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(hpmc_ext, m) {
  using namespace hpmc;

  using StateHS = State<HardSphere, HardSphereParams>;
  using MoveHS = Move<HardSphere>;

  nb::class_<MoveHS>(m, "HardSphereMove").def(nb::init<Vec3>());
  nb::class_<HardSphereParams>(m, "HardSphereParams").def(nb::init<double>());
  nb::class_<HardSphere>(m, "HardSphere")
      .def(nb::init<Vec3>())
      .def_rw("pos", &HardSphere::pos);

  nb::class_<StateHS>(m, "HardSphereEngine")
      .def(nb::init<double, HardSphereParams>(), nb::arg("L"),
           nb::arg("params"))
      .def("size", &StateHS::size)
      .def(
          "get_particles",
          [](StateHS &s) {
            size_t shape[2] = {s.size(), static_cast<size_t>(HardSphere::dof)};

            return View<HardSphere>(
                reinterpret_cast<double *>(
                    const_cast<HardSphere *>(s.get_particles().data())),
                2,     // 2 Dimensions (Rows, DoF columns)
                shape, // Shape configuration
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def(
          "set_particles",
          [](StateHS &s, View<HardSphere> p) {
            s.set_particles(view_to_span<HardSphere>(p));
          },
          nb::arg("particles").noconvert())
      .def("count_overlaps", &StateHS::count_overlaps)
      .def("run_batch", &StateHS::run_batch);
}
