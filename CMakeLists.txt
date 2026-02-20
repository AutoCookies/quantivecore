cmake_minimum_required(VERSION 3.20)
project(quantcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QUANTCORE_ENABLE_ASAN "Enable AddressSanitizer" OFF)

add_library(quantcore
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

target_include_directories(quantcore PUBLIC include)
target_compile_options(quantcore PRIVATE -O3 -Wall -Wextra -Werror)

if(QUANTCORE_ENABLE_ASAN)
    target_compile_options(quantcore PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(quantcore PRIVATE -fsanitize=address)
endif()

enable_testing()

add_executable(quantcore_tests
    tests/test_main.cpp
    tests/test_bitpack.cpp
    tests/test_binary_gemm.cpp
    tests/test_ternary_gemm.cpp
    tests/property_tests.cpp
)
target_link_libraries(quantcore_tests PRIVATE quantcore)
target_compile_options(quantcore_tests PRIVATE -Wall -Wextra -Werror)
if(QUANTCORE_ENABLE_ASAN)
    target_compile_options(quantcore_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(quantcore_tests PRIVATE -fsanitize=address)
endif()

add_test(NAME quantcore_tests COMMAND quantcore_tests)

add_executable(bench_gemm bench/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE quantcore)
target_compile_options(bench_gemm PRIVATE -O3 -march=native -Wall -Wextra -Werror)
if(QUANTCORE_ENABLE_ASAN)
    target_compile_options(bench_gemm PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(bench_gemm PRIVATE -fsanitize=address)
endif()
