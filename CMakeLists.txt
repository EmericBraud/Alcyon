cmake_minimum_required(VERSION 3.14)
project(Chess26 VERSION 1.0 LANGUAGES C CXX)

# --- 1. GESTION DU MODE DE COMPILATION PAR DÉFAUT ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "No build type selected, defaulting to: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 2. OPTIONS ET FLAGS ---
option(ENABLE_GUI "Build with GUI support" OFF)
option(BUILD_TESTING "Build unit tests" OFF)

include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

# Détection des flags CPU (pour x86)
check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
check_cxx_compiler_flag("-mpopcnt" COMPILER_SUPPORTS_POPCNT)

set(CPU_FLAGS -march=native)
if(COMPILER_SUPPORTS_BMI2)
    list(APPEND CPU_FLAGS -mbmi2)
endif()
if(COMPILER_SUPPORTS_POPCNT)
    list(APPEND CPU_FLAGS -mpopcnt)
endif()

# -fno-omit-frame-pointer est vital pour le profiling sur ARM (Mac M1/M2/M3)
# -g1 génère des infos de debug légères pour éviter le bug "DIE" du linker Mac
set(COMMON_FLAGS -Wall -Wextra -fno-omit-frame-pointer ${CPU_FLAGS})
set(PROFILING_FLAGS "-g1")

find_package(Threads REQUIRED)

# --- 3. SOURCES ---
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

if(NOT ENABLE_GUI)
    list(FILTER SOURCES EXCLUDE REGEX "src/interface/gui\\.cpp")
endif()

# --- 4. CIBLES EN MODE 'OBJECT' ---

# Fathom (Tablebases)
add_library(fathom OBJECT lib/fathom/tbprobe.c)
target_include_directories(fathom PUBLIC lib/fathom)
target_compile_definitions(fathom PRIVATE $<$<PLATFORM_ID:Linux>:_REENTRANT>)
target_compile_options(fathom PRIVATE $<$<COMPILE_LANGUAGE:C>:${COMMON_FLAGS} ${PROFILING_FLAGS}>)

# Chess Core
add_library(chess_core OBJECT ${SOURCES})
target_include_directories(chess_core PUBLIC src)
# CORRECTIF : On lie fathom à chess_core pour que src/*.cpp trouve tbprobe.h
target_link_libraries(chess_core PUBLIC fathom) 

target_compile_options(chess_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_FLAGS} ${PROFILING_FLAGS}>)
target_compile_definitions(chess_core PUBLIC NDEBUG SPSA_TUNING)

# --- 5. EXÉCUTABLE PRINCIPAL ---
add_executable(chess26 src/main.cpp)

# On lie les objets et les dépendances
target_link_libraries(chess26 PRIVATE 
    $<TARGET_OBJECTS:chess_core> 
    $<TARGET_OBJECTS:fathom> 
    Threads::Threads
)

# On s'assure que main.cpp trouve aussi les headers
target_include_directories(chess26 PRIVATE src lib/fathom)
target_compile_options(chess26 PRIVATE ${COMMON_FLAGS} ${PROFILING_FLAGS})

if(ENABLE_GUI)
    find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)
    target_link_libraries(chess26 PRIVATE sfml-graphics sfml-window sfml-system)
    target_compile_definitions(chess_core PUBLIC CHESS26_HAS_GUI)
endif()

# --- 6. OPTIMISATIONS AVANCÉES (LTO/IPO) ---
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set_property(TARGET chess26 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "LTO/IPO is enabled and active.")
else()
    message(WARNING "LTO not supported: ${IPO_ERROR}")
endif()

# --- 7. DONNÉES (CORRECTIF BLOQUAGE) ---
set(CHESS26_RUNTIME_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")

add_custom_command(
    TARGET chess26 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CHESS26_RUNTIME_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CHESS26_RUNTIME_DATA_DIR}
    COMMENT "Mise à jour du dossier data dans le build..."
)

# --- 8. TESTS ---
if(BUILD_TESTING)
    # Ta config GTest...
endif()