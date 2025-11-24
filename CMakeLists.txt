cmake_minimum_required(VERSION 3.14)

project(Chess26 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -g)

file(GLOB_RECURSE SOURCES "src/*.cpp")

list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

add_library(chess_core STATIC ${SOURCES})

target_include_directories(chess_core PUBLIC include)


add_executable(chess26 src/main.cpp)
target_link_libraries(chess26 PRIVATE chess_core)

# --- Début des modifications pour créer le dossier 'data' ---

set(CHESS26_DATA_DIR "$<TARGET_FILE_DIR:chess26>/data")
add_custom_command(
    TARGET chess26 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CHESS26_DATA_DIR}
)

# --- Fin des modifications ---

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
set(TEST_SOURCES tests/tests.cpp)
add_executable(chess_tests ${TEST_SOURCES})

target_include_directories(chess_tests PRIVATE 
    include
)

find_package(Threads REQUIRED)
target_link_libraries(chess_tests PRIVATE chess_core gtest_main gtest Threads::Threads)

# --- Début des modifications pour créer le dossier 'data' des tests ---

set(TESTS_DATA_DIR "$<TARGET_FILE_DIR:chess_tests>/data")
add_custom_command(
    TARGET chess_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TESTS_DATA_DIR}
)

# --- Fin des modifications ---

include(GoogleTest)
gtest_discover_tests(chess_tests)