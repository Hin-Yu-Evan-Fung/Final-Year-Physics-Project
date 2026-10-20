cmake_minimum_required(VERSION 3.20)
project(hpmc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v3.1.0
)
find_package(Python 3.14
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)
FetchContent_MakeAvailable(nanobind)

file(GLOB_RECURSE SOURCES
    "src/core/*.cpp"
    "src/particles/*.cpp"
)

include_directories(src)
nanobind_add_module(
  hpmc_ext
  STABLE_ABI
  NB_STATIC
  src/hpmc.cpp
  ${SOURCES}
)
target_compile_definitions(hpmc_ext PRIVATE  $<$<CONFIG:Release>:NDEBUG>)
target_compile_options(hpmc_ext PRIVATE  $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)
set_target_properties(hpmc_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hpmc/"
)

# Google tests

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB_RECURSE GTEST_SOURCES
    "gtest/*.cpp"
)

add_executable(hpmc_tests ${GTEST_SOURCES})

target_include_directories(hpmc_tests PRIVATE src)
target_link_libraries(hpmc_tests PRIVATE gtest_main)
target_compile_definitions(hpmc_tests PRIVATE $<$<CONFIG:Release>:NDEBUG>)
target_compile_options(hpmc_tests PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)

add_test(NAME core_unit_tests COMMAND hpmc_tests)

