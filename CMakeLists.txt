cmake_minimum_required(VERSION 3.14)

project(Chess26 VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_GUI "Build with GUI support" OFF)
option(BUILD_TESTING "Build unit tests" OFF)

include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
check_cxx_compiler_flag("-mpopcnt" COMPILER_SUPPORTS_POPCNT)

set(CPU_FLAGS -march=native)
if(COMPILER_SUPPORTS_BMI2)
    list(APPEND CPU_FLAGS -mbmi2)
endif()
if(COMPILER_SUPPORTS_POPCNT)
    list(APPEND CPU_FLAGS -mpopcnt)
endif()

find_package(Threads REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

if(NOT ENABLE_GUI)
    list(FILTER SOURCES EXCLUDE REGEX "src/interface/gui\\.cpp")
endif()

add_library(chess_core STATIC ${SOURCES})
target_include_directories(chess_core PUBLIC src)

target_compile_options(chess_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O3;-fno-omit-frame-pointer;${CPU_FLAGS}>
)

target_compile_definitions(chess_core PUBLIC
    NDEBUG
    SPSA_TUNING
)

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
    -O3
    -march=native
)

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set_property(TARGET chess_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET chess26 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET fathom PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(WARNING "IPO/LTO not supported: ${IPO_ERROR}")
endif()

set(CHESS26_RUNTIME_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
add_custom_target(copy_runtime_data ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CHESS26_RUNTIME_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CHESS26_RUNTIME_DATA_DIR}
)
add_dependencies(chess26 copy_runtime_data)

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

    if(IPO_SUPPORTED)
        set_property(TARGET chess_tests PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    include(GoogleTest)
    gtest_discover_tests(chess_tests)
endif()