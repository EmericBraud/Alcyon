cmake_minimum_required(VERSION 3.16)
project(plane_batch_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(NNUE_PYTORCH_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../nnue-pytorch/data_loader/cpp")

add_library(plane_batch_loader SHARED
    plane_batch.cpp
    plane_batch_stream.cpp
    plane_batch_abi.cpp
)

target_include_directories(plane_batch_loader PRIVATE
    "${NNUE_PYTORCH_CPP_DIR}"
    "${NNUE_PYTORCH_CPP_DIR}/lib"
)

target_link_libraries(plane_batch_loader PRIVATE Threads::Threads)

if(NOT MSVC)
    target_compile_options(plane_batch_loader PRIVATE -O3 -march=native)
endif()
