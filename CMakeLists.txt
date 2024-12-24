cmake_minimum_required(VERSION 3.21)
project(CLRS_cpp_imp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(GTest REQUIRED)
include_directories(${GTest_INCLUDE_DIRS})

find_package(benchmark REQUIRED)


set(TEST_SOURCES
    source/test.cpp
)

set(BENCHMARK_SOURCES
    source/benchmark.cpp
)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE
    GTest::GTest GTest::Main
)

add_executable(benchmark ${BENCHMARK_SOURCES})
target_link_libraries(benchmark 
    PRIVATE
    benchmark::benchmark
)
