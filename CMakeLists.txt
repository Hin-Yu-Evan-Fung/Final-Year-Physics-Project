cmake_minimum_required(VERSION 3.20)
project(hsmc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "Use pip install .")
endif()

find_package(Python 3.14
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES
    "src/core/*.cpp"
    "src/particles/*.cpp"
)

include_directories(src)
nanobind_add_module(
  hsmc
  src/hsmc.cpp
  ${SOURCES}
)
target_compile_definitions(hsmc PRIVATE  $<$<CONFIG:Release>:NDEBUG>)
target_compile_options(hsmc PRIVATE  $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)
install(TARGETS hsmc DESTINATION .)

find_program(NANOBIND_STUBGEN NAMES nanobind-stubgen)

if(NOT CMAKE_SKIP_INSTALL_RULES)
  nanobind_add_stub(
      hsmc_stub
      MODULE hsmc
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/python/hsmc.pyi"
      DEPENDS hsmc
    )
endif()

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

add_executable(
  hsmc_tests 
  ${GTEST_SOURCES}
)

target_include_directories(hsmc_tests PRIVATE src)
target_link_libraries(hsmc_tests PRIVATE gtest_main)
target_compile_definitions(hsmc_tests PRIVATE $<$<CONFIG:Release>:NDEBUG>)
target_compile_options(hsmc_tests PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math>)

add_test(NAME core_unit_tests COMMAND hsmc_tests)
