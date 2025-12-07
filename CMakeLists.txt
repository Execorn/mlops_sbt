cmake_minimum_required(VERSION 3.15)
project(tensor_ops LANGUAGES C CXX)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
include(GNUInstallDirs)

pybind11_add_module(_tensor_ops src/main.cpp)

target_link_libraries(_tensor_ops PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(_tensor_ops PRIVATE /O2 /fp:fast)
else()
    target_compile_options(_tensor_ops PRIVATE -O3 -march=native -ffast-math)
endif()

install(TARGETS _tensor_ops
        LIBRARY DESTINATION tensor_ops
        RUNTIME   DESTINATION tensor_ops)