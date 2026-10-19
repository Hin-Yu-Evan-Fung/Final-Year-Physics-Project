#include <nanobind/nanobind.h>

NB_MODULE(hsmc, m) {
  m.def("hello", []() { return "Hello world!"; });
}
