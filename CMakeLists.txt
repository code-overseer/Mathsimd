cmake_minimum_required(VERSION 3.10)
project(Mathematics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O1 -pthread -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")

if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.14")
endif()

add_executable(Mathematics 
    main.cpp
    include/policies/M128.hpp
    tests/tests.hpp
    tests/tests.cpp )
