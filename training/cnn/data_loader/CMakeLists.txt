cmake_minimum_required(VERSION 3.16)
project(plane_batch_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(NNUE_PYTORCH_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../nnue-pytorch/data_loader/cpp")
set(CHESS26_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# Reuse the real engine's build (chess_core object library) to get NNUE
# eval + board + move-generation code — see nnue_bridge.cpp. Building
# against the exact same objects used by the tested `chess26` binary is
# far less error-prone than hand-picking a minimal file subset, at the
# cost of some unused compile time (search/UCI/tablebase code compiled
# but never linked into anything we invoke).
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_GUI OFF CACHE BOOL "" FORCE)
add_subdirectory("${CHESS26_ROOT_DIR}" chess26_engine_build)

add_library(plane_batch_loader SHARED
    plane_batch.cpp
    plane_batch_stream.cpp
    plane_batch_abi.cpp
    nnue_bridge.cpp
)

target_include_directories(plane_batch_loader PRIVATE
    "${NNUE_PYTORCH_CPP_DIR}"
    "${NNUE_PYTORCH_CPP_DIR}/lib"
    "${CHESS26_ROOT_DIR}/src"
    "${CHESS26_ROOT_DIR}/lib/fathom"
)

target_link_libraries(plane_batch_loader PRIVATE
    Threads::Threads
    $<TARGET_OBJECTS:chess_core>
    $<TARGET_OBJECTS:fathom>
)

if(NOT MSVC)
    target_compile_options(plane_batch_loader PRIVATE -O3 -march=native)
endif()

# One-off debugging tool: dumps FEN + game result for a validation-split
# sample, so NNUE (FEN-only via UCI) can be checked against real game
# outcomes the same way the CNN already is via PlaneBatchDataset tensors.
# Also compiles plane_batch.cpp (for hash_position), hence the same NNUE
# engine dependencies as the main library above.
add_executable(dump_val_fens dump_val_fens.cpp plane_batch.cpp nnue_bridge.cpp)
target_include_directories(dump_val_fens PRIVATE
    "${NNUE_PYTORCH_CPP_DIR}"
    "${NNUE_PYTORCH_CPP_DIR}/lib"
    "${CHESS26_ROOT_DIR}/src"
    "${CHESS26_ROOT_DIR}/lib/fathom"
)
target_link_libraries(dump_val_fens PRIVATE
    Threads::Threads
    $<TARGET_OBJECTS:chess_core>
    $<TARGET_OBJECTS:fathom>
)
if(NOT MSVC)
    target_compile_options(dump_val_fens PRIVATE -O3 -march=native)
endif()
