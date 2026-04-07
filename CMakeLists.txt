cmake_minimum_required(VERSION 3.14)

project(Chess26 VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GUI support (off by default for OpenBench)
option(ENABLE_GUI "Build with GUI support" OFF)
option(BUILD_TESTING "Build unit tests" OFF)

# Check for supported compiler flags
include(CheckCXXCompilerFlag)

# Test if CPU flags are supported
check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
check_cxx_compiler_flag("-mpopcnt" COMPILER_SUPPORTS_POPCNT)

# Build the CPU flags string dynamically
set(CPU_FLAGS "-march=native")
if(COMPILER_SUPPORTS_BMI2)
    set(CPU_FLAGS "${CPU_FLAGS};-mbmi2")
endif()
if(COMPILER_SUPPORTS_POPCNT)
    set(CPU_FLAGS "${CPU_FLAGS};-mpopcnt")
endif()

# -------------------------------------------------------------------
# Threads (needed for Threads::Threads)
# -------------------------------------------------------------------
find_package(Threads REQUIRED)

# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

# Exclude GUI if not enabled
if(NOT ENABLE_GUI)
    list(FILTER SOURCES EXCLUDE REGEX "src/interface/gui\\.cpp")
endif()

add_library(chess_core STATIC ${SOURCES})
target_include_directories(chess_core PUBLIC src)

target_compile_options(chess_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-flto;-O3;-DNDEBUG;${CPU_FLAGS};-fno-omit-frame-pointer;-DSPSA_TUNING>
)

# -------------------------------------------------------------------
# SFML (optional, only if GUI is enabled)
# -------------------------------------------------------------------
if(ENABLE_GUI)
    find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)
    target_link_libraries(chess_core PUBLIC
        sfml-graphics
        sfml-window
        sfml-system
        Threads::Threads
    )
    target_compile_definitions(chess_core PUBLIC CHESS26_HAS_GUI)
else()
    target_link_libraries(chess_core PUBLIC Threads::Threads)
endif()

# -------------------------------------------------------------------
# Fathom
# -------------------------------------------------------------------
add_library(fathom STATIC lib/fathom/tbprobe.c)

target_compile_definitions(fathom PRIVATE 
    $<$<PLATFORM_ID:Linux>:_REENTRANT>
)

target_link_libraries(fathom PRIVATE Threads::Threads)

set_target_properties(fathom PROPERTIES
    LINKER_LANGUAGE C
)

target_include_directories(fathom PUBLIC
    lib/fathom
)

target_compile_options(fathom PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3;${CPU_FLAGS};-fno-omit-frame-pointer>
)

target_link_libraries(chess_core PUBLIC fathom)

# -------------------------------------------------------------------
# Executable chess26
# -------------------------------------------------------------------
add_executable(chess26 src/main.cpp)

if(ENABLE_GUI)
    target_link_libraries(chess26 PRIVATE
        chess_core
        sfml-graphics
        sfml-window
        sfml-system
    )
else()
    target_link_libraries(chess26 PRIVATE chess_core)
endif()

target_link_options(chess26 PRIVATE 
    $<$<CONFIG:Release>:-O3;-flto;-march=native>
    $<$<NOT:$<CONFIG:Release>>:-O3;-flto;-march=native>
)

set(CHESS26_RUNTIME_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
add_custom_target(copy_runtime_data ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CHESS26_RUNTIME_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CHESS26_RUNTIME_DATA_DIR}
)
add_dependencies(chess26 copy_runtime_data)

# -------------------------------------------------------------------
# GoogleTest
# -------------------------------------------------------------------
if(BUILD_TESTING)
    cmake_policy(SET CMP0135 NEW)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
    add_executable(chess_tests ${TEST_SOURCES})

    target_include_directories(chess_tests PRIVATE include)

    target_link_libraries(chess_tests PRIVATE
        chess_core
        gtest_main
        gtest
        Threads::Threads
    )

    add_dependencies(chess_tests copy_runtime_data)

    include(GoogleTest)
    gtest_discover_tests(chess_tests)
endif()

