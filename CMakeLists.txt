
cmake_minimum_required(VERSION 3.20)
project(hsmc LANGUAGES CXX)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "Use pip install .")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.14
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(hsmc STABLE_ABI src/hsmc.cpp)

install(TARGETS hsmc LIBRARY DESTINATION .)

find_program(NANOBIND_STUBGEN NAMES nanobind-stubgen)

if(NOT CMAKE_SKIP_INSTALL_RULES)
  nanobind_add_stub(
      hsmc_stub
      MODULE hsmc
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/python/hsmc.pyi"
      DEPENDS hsmc
    )
endif()
