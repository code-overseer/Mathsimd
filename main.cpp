#include <cstdio>
#include "tests/tests.hpp"

int main() {

    for (auto i = 0; i < 1000; ++i) {
        mathtests::test_float2_dot();
        mathtests::test_float3_dot();
        mathtests::test_float4_dot();
        mathtests::test_float3_cross();
        mathtests::test_float4_cross();
        mathtests::test_float4x4_matmul();
        mathtests::test_float4x4_vecmul();
    }



    return 0;
}