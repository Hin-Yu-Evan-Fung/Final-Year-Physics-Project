cmake_minimum_required(VERSION 3.20)
project(hpmc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# -----------------------------------------------------------------------------
# Dependencies: nanobind & Python
# -----------------------------------------------------------------------------

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v3.1.0
)
find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)
FetchContent_MakeAvailable(nanobind)

# -----------------------------------------------------------------------------
# Target: hpmc_ext (Python Module)
# -----------------------------------------------------------------------------

set(SOURCES
  src/core/state.cpp
  src/single_engine.cpp
)

nanobind_add_module(
  hpmc_ext
  STABLE_ABI
  NB_STATIC
  src/hpmc.cpp
  ${SOURCES}
)

target_include_directories(hpmc_ext PRIVATE src)

# Target-scoped Compiler Flags (Handles Clang vs GCC for macOS/Linux)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Apple Clang (macOS) & LLVM Clang (Linux/Windows-MinGW)
  target_compile_options(hpmc_ext PRIVATE $<$<CONFIG:Release>:-g -O3>)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # Pure GCC optimizations
  target_compile_options(hpmc_ext PRIVATE $<$<CONFIG:Release>:-g -O3 -march=native -ffast-math -fopt-info-vec-optimized>)
endif()

set_target_properties(hpmc_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hpmc/"
)

nanobind_add_stub(
  hpmc_ext_stub
  MODULE hpmc_ext
  TARGET hpmc_ext
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/hpmc/hpmc_ext.pyi"
  PYTHON_PATH $<TARGET_FILE_DIR:hpmc_ext>
  DEPENDS hpmc_ext
)

# -----------------------------------------------------------------------------
# Target: Google Test
# -----------------------------------------------------------------------------

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

add_executable(hpmc_tests ${SOURCES} ${GTEST_SOURCES})

target_include_directories(hpmc_tests PRIVATE src)
target_link_libraries(hpmc_tests PRIVATE gtest_main)
target_compile_definitions(hpmc_tests PRIVATE $<$<CONFIG:Release>:NDEBUG>)
target_compile_options(hpmc_tests PRIVATE $<$<CONFIG:Release>:-O3>)

add_test(NAME core_unit_tests COMMAND hpmc_tests)

