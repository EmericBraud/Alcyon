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

# One-off debugging tool: dumps FEN + game result for a validation-split
# sample, so NNUE (FEN-only via UCI) can be checked against real game
# outcomes the same way the CNN already is via PlaneBatchDataset tensors.
add_executable(dump_val_fens dump_val_fens.cpp plane_batch.cpp)
target_include_directories(dump_val_fens PRIVATE
    "${NNUE_PYTORCH_CPP_DIR}"
    "${NNUE_PYTORCH_CPP_DIR}/lib"
)
target_link_libraries(dump_val_fens PRIVATE Threads::Threads)
if(NOT MSVC)
    target_compile_options(dump_val_fens PRIVATE -O3 -march=native)
endif()
