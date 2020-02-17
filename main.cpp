#include <cstdio>
#include "include/tests.hpp"

int main() {
    printf("Begin tests...\n");
    mathtests::test_seq_dot();
    mathtests::test_simd_dot();
//    mathtests::test_seq_cross();
    mathtests::test_simd_cross();
    printf("All tests passed!\n");

    printf("Begin benchmarks...\n");
//    mathtests::benchmark_seq_dot();
    mathtests::benchmark_simd_dot();

    printf("Done\n");
    return 0;
}