cmake_minimum_required(VERSION 3.14)

project(Chess26 VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CHESS26_DATA_FULL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data/")
add_definitions(-DDATA_PATH="${CHESS26_DATA_FULL_PATH}")


#add_compile_options(-Wall -Wextra -DNDEBUG -flto -O3 -march=native -mbmi2 -fno-omit-frame-pointer -mpopcnt -fprofile-generate -lpthread)
#add_compile_options(-Wall -Wextra -flto -O3 -march=native -ggdb3 -fno-omit-frame-pointer -mbmi2 -mpopcnt)
#add_compile_options(-Wall -Wextra -march=native -g -ggdb3 -fno-omit-frame-pointer -mbmi2 -O2)

#add_link_options(-fprofile-use)
#add_link_options(-fprofile-generate)





# -------------------------------------------------------------------
# Threads (needed for Threads::Threads)
# -------------------------------------------------------------------
find_package(Threads REQUIRED)

# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

add_library(chess_core STATIC ${SOURCES})
target_include_directories(chess_core PUBLIC src)

target_compile_options(chess_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-flto;-O3;-DNDEBUG;-march=native;-mbmi2;-fno-omit-frame-pointer;-mpopcnt;-DSPSA_TUNING>
)

 
# -------------------------------------------------------------------
# SFML
# -------------------------------------------------------------------
find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)

target_link_libraries(chess_core PUBLIC
    sfml-graphics
    sfml-window
    sfml-system
    Threads::Threads
)

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

find_package(Threads REQUIRED)

target_compile_definitions(fathom PRIVATE 
    # On peut définir explicitement qu'on veut les threads 
    # (ou s'assurer que TB_NO_THREADS n'est PAS là)
    $<$<PLATFORM_ID:Linux>:_REENTRANT>
)

target_link_libraries(fathom PRIVATE Threads::Threads)

target_link_libraries(chess_core PUBLIC fathom)

target_compile_options(fathom PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3;-march=native;-mbmi2;-mpopcnt;-fno-omit-frame-pointer>
)

# -------------------------------------------------------------------
# Executable chess26
# -------------------------------------------------------------------
add_executable(chess26 src/main.cpp)
target_link_libraries(chess26 PRIVATE
    chess_core
    sfml-graphics
    sfml-window
    sfml-system
)

target_link_options(chess26 PRIVATE 
    $<$<CONFIG:Release>:-O3;-flto;-march=native>
    $<$<NOT:$<CONFIG:Release>>:-O3;-flto;-march=native>
)
set(CHESS26_DATA_DIR "$<TARGET_FILE_DIR:chess26>/data")
add_custom_command(
    TARGET chess26 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CHESS26_DATA_DIR}
)

# -------------------------------------------------------------------
# GoogleTest
# -------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
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

set(TESTS_DATA_DIR "$<TARGET_FILE_DIR:chess_tests>/data")
add_custom_command(
    TARGET chess_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TESTS_DATA_DIR}
)

include(GoogleTest)
gtest_discover_tests(chess_tests)

