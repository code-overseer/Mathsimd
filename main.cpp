#include <cstdio>
#include "include/tests.hpp"
#include "include/simd/float3.hpp"

int main() {
    printf("Begin tests...\n");
    mathtests::test_seq_dot();
    mathtests::test_simd_dot();
//    mathtests::test_seq_cross();
    mathtests::test_simd_cross();
    printf("All tests passed!\n");

    printf("Begin benchmarks...\n");
    mathtests::benchmark_seq_dot();
    mathtests::benchmark_simd_dot();

    mathtests::benchmark_seq_cross();
    mathtests::benchmark_simd_cross();

    printf("Done\n");

    printf("%lu\n", sizeof(mathsimd::float3));
    mathsimd::float3 abc{1,5,-2};
    abc = abc/1.2;
    printf("%f,%f,%f\n",abc.x(),abc.y(),abc.z());

    return 0;
}