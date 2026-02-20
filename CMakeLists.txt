cmake_minimum_required(VERSION 3.20)
project(quantcore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QUANTCORE_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(QUANTCORE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(QUANTCORE_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(QC_ENABLE_AVX512 "Enable AVX-512 dispatch paths" ON)
option(QC_ENABLE_AMX "Enable AMX dispatch paths" ON)
option(QC_REPRODUCIBLE "Enable deterministic/reproducible build settings" OFF)
option(QC_BUILD_EXAMPLES "Build deployment examples" ON)

set(QC_SAN_FLAGS "")
if(QUANTCORE_ENABLE_ASAN)
  string(APPEND QC_SAN_FLAGS " -fsanitize=address")
endif()
if(QUANTCORE_ENABLE_UBSAN)
  string(APPEND QC_SAN_FLAGS " -fsanitize=undefined")
endif()
if(QUANTCORE_ENABLE_TSAN)
  string(APPEND QC_SAN_FLAGS " -fsanitize=thread")
endif()
if(NOT QC_SAN_FLAGS STREQUAL "")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${QC_SAN_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${QC_SAN_FLAGS}")
endif()

set(QC_SOURCES
    src/bitpack.cpp
    src/binary_gemm.cpp
    src/ternary_gemm.cpp
    src/format.cpp
    src/binary_gemm_avx2.cpp
    src/ternary_gemm_avx2.cpp
    src/binary_gemm_avx512.cpp
    src/ternary_gemm_avx512.cpp
    src/binary_gemm_amx.cpp
    src/blocking_strategy.cpp
    src/numa_threadpool.cpp
    src/perf_counters.cpp
    src/dispatch.cpp
)

add_library(quantcore_obj OBJECT ${QC_SOURCES})
target_include_directories(quantcore_obj PUBLIC include)
target_compile_definitions(quantcore_obj PUBLIC QC_ENABLE_AVX512=$<BOOL:${QC_ENABLE_AVX512}> QC_ENABLE_AMX=$<BOOL:${QC_ENABLE_AMX}>)
target_compile_options(quantcore_obj PRIVATE -O3 -Wall -Wextra -Werror -Wpedantic -fstack-protector-strong -D_FORTIFY_SOURCE=2)
set_target_properties(quantcore_obj PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

if(QUANTCORE_ENABLE_ASAN)
    target_compile_options(quantcore_obj PRIVATE -fsanitize=address -fno-omit-frame-pointer)
endif()
if(QUANTCORE_ENABLE_UBSAN)
    target_compile_options(quantcore_obj PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
endif()
if(QUANTCORE_ENABLE_TSAN)
    target_compile_options(quantcore_obj PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
endif()

if(QC_REPRODUCIBLE)
    target_compile_options(quantcore_obj PRIVATE -ffile-prefix-map=${CMAKE_SOURCE_DIR}=. -ffile-prefix-map=${CMAKE_BINARY_DIR}=.)
    target_compile_definitions(quantcore_obj PRIVATE SOURCE_DATE_EPOCH=$ENV{SOURCE_DATE_EPOCH})
endif()

add_library(quantcore_static STATIC $<TARGET_OBJECTS:quantcore_obj>)
add_library(quantcore_shared SHARED $<TARGET_OBJECTS:quantcore_obj>)
set_target_properties(quantcore_static PROPERTIES OUTPUT_NAME quantcore)
set_target_properties(quantcore_shared PROPERTIES OUTPUT_NAME quantcore SOVERSION 1 VERSION ${PROJECT_VERSION})
target_include_directories(quantcore_static PUBLIC include)
target_include_directories(quantcore_shared PUBLIC include)
target_compile_definitions(quantcore_shared PRIVATE QUANTCORE_BUILD_SHARED)
if(QC_REPRODUCIBLE)
    target_link_options(quantcore_shared PRIVATE -Wl,--build-id=none)
endif()

add_library(quantcore ALIAS quantcore_shared)

enable_testing()

add_executable(quantcore_tests
    tests/test_main.cpp
    tests/test_bitpack.cpp
    tests/test_binary_gemm.cpp
    tests/test_ternary_gemm.cpp
    tests/property_tests.cpp
    tests/test_integration.cpp
)
target_link_libraries(quantcore_tests PRIVATE quantcore_static)
target_compile_options(quantcore_tests PRIVATE -Wall -Wextra -Werror -Wpedantic)
if(QUANTCORE_ENABLE_ASAN)
    target_compile_options(quantcore_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(quantcore_tests PRIVATE -fsanitize=address)
endif()
if(QUANTCORE_ENABLE_UBSAN)
    target_compile_options(quantcore_tests PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(quantcore_tests PRIVATE -fsanitize=undefined)
endif()
if(QUANTCORE_ENABLE_TSAN)
    target_compile_options(quantcore_tests PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(quantcore_tests PRIVATE -fsanitize=thread)
endif()
add_test(NAME quantcore_tests COMMAND quantcore_tests)

add_executable(bench_gemm bench/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE quantcore_static)
target_compile_options(bench_gemm PRIVATE -O3 -march=native -Wall -Wextra -Werror -Wpedantic)

install(TARGETS quantcore_static quantcore_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(FILES include/quantcore/c_api.h include/quantcore/version.hpp DESTINATION include/quantcore)

configure_file(${CMAKE_SOURCE_DIR}/quantcore.pc.in ${CMAKE_BINARY_DIR}/quantcore.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/quantcore.pc DESTINATION lib/pkgconfig)

if(QC_BUILD_EXAMPLES)
  add_executable(simple_inference examples/simple_inference.cpp)
  target_link_libraries(simple_inference PRIVATE quantcore_shared)
  target_compile_options(simple_inference PRIVATE -Wall -Wextra -Werror -Wpedantic)
endif()
