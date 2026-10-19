#include "core/state.hpp"
#include "core/vec3.hpp"
#include "particles/hard_sphere.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // Modern explicit STL headers

namespace nb = nanobind;

NB_MODULE(hsmc, m) {
  using namespace hsmc;

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
      .def("set_particles", &StateHS::set_particles)
      .def("try_run_batch", &StateHS::try_run_batch,
           nb::arg("indices").noconvert(), nb::arg("moves").noconvert())
      .def("run_batch", &StateHS::run_batch, nb::arg("indices").noconvert(),
           nb::arg("moves").noconvert());
}
