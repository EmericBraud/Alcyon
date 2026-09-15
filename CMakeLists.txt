cmake_minimum_required(VERSION 3.14)
project(Chess26 VERSION 1.0 LANGUAGES C CXX)

# --- 1. GESTION DU MODE DE COMPILATION PAR DÉFAUT ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "No build type selected, defaulting to: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 2. OPTIONS ET FLAGS ---
option(ENABLE_GUI "Build with GUI support" OFF)
option(BUILD_TESTING "Build unit tests" ON)
option(ENABLE_GPU_EVAL "Enable the native GPU (Metal) CNN eval queue -- see src/engine/eval/gpu/" OFF)
# Interrupteurs de mesure de la recherche (histogramme des rangs de coupure,
# desactivation des coupures TT pour l'experience d'oracle d'ordonnancement).
# Off par defaut : ils sont testes sur le chemin chaud et ne servent qu'a
# mesurer. Voir namespace search dans src/engine/search/worker.hpp.
option(ENABLE_SEARCH_EXPERIMENTS "Build the search measurement switches (CHESS26_ORDER_STATS, CHESS26_TT_NO_CUTOFF)" OFF)

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
set(COMMON_FLAGS -Wall -Wextra -fno-omit-frame-pointer -fno-exceptions ${CPU_FLAGS})
set(PROFILING_FLAGS "-g1")

find_package(Threads REQUIRED)

# --- 3. SOURCES ---
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main.cpp")

if(ENABLE_GPU_EVAL)
    if(NOT APPLE)
        message(FATAL_ERROR "ENABLE_GPU_EVAL currently only supports Apple/Metal")
    endif()
    # metal_backend.mm (Objective-C++, ARC) needs Apple Clang specifically
    # -- GCC's Objective-C++ support doesn't accept -fobjc-arc at all
    # (tested: Homebrew g++ rejects it outright). The REST of chess_core
    # can't switch to Apple Clang either way: accumulator_layer.hpp uses
    # GCC's <experimental/simd> extension, which doesn't exist in Apple
    # Clang's libc++. So this forces the OBJCXX compiler ONLY (leaving
    # CMAKE_CXX_COMPILER, and everything else, as the user configured it)
    # -- meaning metal_backend.mm's object file is linked in from a
    # different C++ toolchain/standard library than the rest of the
    # binary. That's only safe because the boundary between them
    # (gpu_backend.hpp) is a plain extern "C" POD-only interface -- see
    # that header's comment before changing it.
    if(NOT DEFINED CMAKE_OBJCXX_COMPILER)
        find_program(APPLE_CLANGXX_FOR_GPU_EVAL clang++ PATHS /usr/bin NO_DEFAULT_PATH)
        if(NOT APPLE_CLANGXX_FOR_GPU_EVAL)
            message(FATAL_ERROR "ENABLE_GPU_EVAL needs Apple Clang for metal_backend.mm -- /usr/bin/clang++ not found")
        endif()
        set(CMAKE_OBJCXX_COMPILER "${APPLE_CLANGXX_FOR_GPU_EVAL}")
    endif()
    enable_language(OBJCXX)
    if(NOT CMAKE_OBJCXX_COMPILER_ID STREQUAL "AppleClang")
        message(FATAL_ERROR "ENABLE_GPU_EVAL's OBJCXX compiler must be Apple Clang (got ${CMAKE_OBJCXX_COMPILER_ID}) -- "
                             "reconfigure with -DCMAKE_OBJCXX_COMPILER=/usr/bin/clang++")
    endif()
    # metal_backend.mm (Objective-C++) replaces the plain-C++ no-op stub.
    list(FILTER SOURCES EXCLUDE REGEX "src/engine/eval/gpu/gpu_backend_stub\\.cpp")
endif()

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

option(ENABLE_TEXEL_TUNING "Enable Texel tuning mode" OFF)
option(ENABLE_SPSA_TUNING "Enable SPSA tuning mode" OFF)
option(ENABLE_NNUE_EVAL "Enable NNUE evaluation" ON)

target_compile_definitions(chess_core PUBLIC NDEBUG)

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




if(ENABLE_TEXEL_TUNING)
    target_compile_definitions(chess_core PUBLIC TEXEL_TUNING)
    target_compile_definitions(chess26 PRIVATE TEXEL_TUNING)
endif()

if(ENABLE_SPSA_TUNING)
    target_compile_definitions(chess_core PUBLIC SPSA_TUNING)
    target_compile_definitions(chess26 PRIVATE SPSA_TUNING)
endif()

if(ENABLE_NNUE_EVAL)
    target_compile_definitions(chess_core PUBLIC NNUE_EVAL)
    target_compile_definitions(chess26 PRIVATE NNUE_EVAL)
endif()

if(ENABLE_GPU_EVAL)
    target_sources(chess_core PRIVATE
        src/engine/eval/gpu/metal_backend.mm
        src/engine/eval/gpu/coreml_backend.mm
    )
    set_source_files_properties(
        src/engine/eval/gpu/metal_backend.mm
        src/engine/eval/gpu/coreml_backend.mm
        PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        LANGUAGE OBJCXX
    )
    target_compile_definitions(chess_core PUBLIC CHESS26_GPU_EVAL_METAL)
    target_compile_definitions(chess26 PRIVATE CHESS26_GPU_EVAL_METAL)
endif()

if(ENABLE_SEARCH_EXPERIMENTS)
    target_compile_definitions(chess_core PUBLIC CHESS26_SEARCH_EXPERIMENTS)
    target_compile_definitions(chess26 PRIVATE CHESS26_SEARCH_EXPERIMENTS)
endif()

if(ENABLE_GPU_EVAL)

    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(MPS_GRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    # chess26 pulls chess_core's object files via $<TARGET_OBJECTS:...>,
    # not a target_link_libraries(chess26 ... chess_core) link -- usage
    # requirements (PUBLIC link libraries, compile definitions) attached
    # to an OBJECT library do NOT propagate through TARGET_OBJECTS, so
    # these frameworks (and CHESS26_GPU_EVAL_METAL above) must be added
    # to chess26 explicitly too, same as Threads::Threads already is.
    # metal_backend.mm's OWN internal std:: usage (std::ifstream, std::
    # vector, ...) needs libc++ (Apple Clang's default), while the rest
    # of the binary is built with whatever standard library the
    # project's normal compiler defaults to (typically libstdc++ for
    # GCC) -- both runtime libraries coexist fine in one binary since
    # their C++ symbols are mangled under different inline namespaces
    # (libc++'s std::__1:: vs libstdc++'s std::__cxx11::/plain std::),
    # so there's no symbol collision; only the extern "C" boundary
    # (gpu_backend.hpp) needs to (and does) stay POD-only.
    target_link_libraries(chess_core PUBLIC ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${MPS_GRAPH_FRAMEWORK} ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK} c++)
    target_link_libraries(chess26 PRIVATE ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${MPS_GRAPH_FRAMEWORK} ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK} c++)
endif()

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
include(CTest)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )

    # Évite d'écraser les settings du parent sur MSVC
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

    add_executable(chess26_tests ${TEST_SOURCES})

    target_link_libraries(chess26_tests PRIVATE
        $<TARGET_OBJECTS:chess_core>
        $<TARGET_OBJECTS:fathom>
        Threads::Threads
        GTest::gtest_main
    )

    target_include_directories(chess26_tests PRIVATE
        src
        lib/fathom
        tests
    )

    target_compile_definitions(chess26_tests PRIVATE CHESS26_UNIT_TESTING)

    # Meme raison que pour chess26 (voir la section ENABLE_GPU_EVAL) : les
    # usage requirements de chess_core ne traversent pas $<TARGET_OBJECTS:>,
    # donc la cible de test doit redeclarer frameworks et definitions. Sans
    # ca, "cmake --build" echouait au lien sur _MTLCreateSystemDefaultDevice
    # et _OBJC_CLASS_$_MLModel des que ENABLE_GPU_EVAL etait on.
    if(ENABLE_GPU_EVAL)
        target_compile_definitions(chess26_tests PRIVATE CHESS26_GPU_EVAL_METAL)
        target_link_libraries(chess26_tests PRIVATE ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${MPS_GRAPH_FRAMEWORK} ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK} c++)
    endif()

    if(ENABLE_SEARCH_EXPERIMENTS)
        target_compile_definitions(chess26_tests PRIVATE CHESS26_SEARCH_EXPERIMENTS)
    endif()
    # Enable NNUE_EVAL for tests only when NNUE support is enabled for the main build
    if(ENABLE_NNUE_EVAL)
        target_compile_definitions(chess26_tests PRIVATE NNUE_EVAL)
    endif()

    target_compile_options(chess26_tests PRIVATE ${COMMON_FLAGS} ${PROFILING_FLAGS})

    if(ENABLE_TEXEL_TUNING)
        target_compile_definitions(chess26_tests PRIVATE TEXEL_TUNING)
    endif()

    if(ENABLE_SPSA_TUNING)
        target_compile_definitions(chess26_tests PRIVATE SPSA_TUNING)
    endif()

    if(ENABLE_GUI)
        target_compile_definitions(chess26_tests PRIVATE CHESS26_HAS_GUI)
    endif()

    include(GoogleTest)
    gtest_discover_tests(chess26_tests)
endif()
