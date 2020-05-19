#ifndef MATHEMATICS_TESTS_HPP
#define MATHEMATICS_TESTS_HPP
#include "../include/mathsimd.hpp"
#include <array>
#include <ctime>
#include <cmath>
#include <iostream>
#include <chrono>

namespace mathtests {

    void test_float3_dot();

    void test_float3_cross();

    void test_float2_dot();

    void test_float2_sign();

    void test_float4_dot();

    void test_float4x4_matmul();
    void test_float4x4_vecmul();

    void test_float4_cross();

    void benchmark_simd_dot();

    void benchmark_simd_cross();


}

#endif //MATHEMATICS_TESTS_HPP
