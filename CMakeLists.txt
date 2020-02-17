cmake_minimum_required(VERSION 3.15)
project(Mathematics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -pthread -march=native -ffast-math -mavx512f")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")


add_executable(Mathematics
        main.cpp
        src/seq/float3.cpp
        src/simd/float3.cpp src/tests.cpp)