#include "../include/mathematics.hpp"
#include "../include/tests.hpp"
#include <array>
#include <ctime>
#include <cmath>
#include <iostream>
#include <chrono>


constexpr unsigned int TESTS = 1000000u;
constexpr unsigned int VALUES = 100u;

static float rnd() {
    static int seed = static_cast<int>(std::time(nullptr));
    seed = int(std::fmod(static_cast<float>(seed) * 1373.f + 691.f, 509.f));
    return static_cast<float>(seed) / 509.f;
}

static unsigned int rnd_idx() {
    return static_cast<unsigned int>(std::time(nullptr)) % VALUES;
}

void mathtests::test_simd_dot() {
    using namespace mathsimd;
    float3 a(1,3,5),b(2,4,6);
    assert(float3::dot(a,b) == 2.f + 12.f + 30.f);
}

void mathtests::test_seq_dot() {
    using namespace mathseq;
    float3 a(1,3,5),b(2,4,6);
    assert(float3::dot(a,b) == 2.f + 12.f + 30.f);
}

void mathtests::test_simd_cross() {
    using namespace mathsimd;
    float3 a(1,3,5);
    float3 b(2,4,6);
    float3 expected(3.f * 6.f - 5.f * 4.f,
           2.f*5.f - 1.f*6.f,
           1.f*4.f - 2.f*3.f);
    auto actual =float3::cross(a, b);
    printf("%f,%f,%f\n",expected.x(),expected.y(),expected.z());
    printf("%f,%f,%f\n",actual.x(),actual.y(),actual.z());
    assert(actual == expected);
}

void mathtests::test_seq_cross() {
    using namespace mathseq;
    float3 a(1,3,5),b(2,4,6);
    float3 expected(3.f * 6.f - 5.f * 4.f,
               2.f*5.f - 1.f*6.f,
               1.f*4.f - 2.f*3.f);
    auto actual = float3::cross(a,b);
    assert(actual == expected);
}

static std::array<mathsimd::float3,VALUES>& generate_simd_vectors() {
    static bool created = false;
    static std::array<mathsimd::float3,VALUES> test_cases;
    if (created) return test_cases;
    for (auto &val : test_cases) {
        val = mathsimd::float3(rnd(),rnd(),rnd());
    }
    created = true;
    return test_cases;
}

static std::array<mathseq::float3,VALUES>& generate_seq_vectors() {
    static bool created = false;
    static std::array<mathseq::float3,VALUES> test_cases;
    if (created) return test_cases;
    for (auto &val : test_cases) { val = mathseq::float3{rnd(),rnd(),rnd() }; }
    created = true;
    return test_cases;
}

void mathtests::benchmark_simd_dot() {
    auto& vectors = generate_simd_vectors();
    using namespace std::chrono;
    float c = 0.f;
    auto start = high_resolution_clock::now();
    for (auto i = 0ll; i < TESTS; ++i) {
        c += mathsimd::float3::dot(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    auto elapsed = high_resolution_clock::now() - start;
    printf("Dot SIMD: %f\n", static_cast<double>(duration_cast<microseconds>(elapsed).count()) / TESTS);
    std::cout<<"Value of sum is "<<c<<std::endl;
}

void mathtests::benchmark_seq_dot() {
    auto& vectors = generate_seq_vectors();
    using namespace std::chrono;
    float c = 0.f;
    auto start = high_resolution_clock::now();
    for (auto i = 0ll; i < TESTS; ++i) {
        c += mathseq::float3::dot(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    auto elapsed = high_resolution_clock::now() - start;
    printf("Dot SEQ: %f\n", static_cast<double>(duration_cast<microseconds>(elapsed).count()) / TESTS);
    std::cout<<"Value of sum is "<<c<<std::endl;
}

void mathtests::benchmark_simd_cross() {
    auto& vectors = generate_simd_vectors();
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (auto i = 0ll; i < TESTS; ++i) {
        vectors[i % VALUES] = mathsimd::float3::cross(vectors[rnd_idx()],vectors[rnd_idx()]);

    }
    auto elapsed = high_resolution_clock::now() - start;
    printf("Cross SIMD: %f\n", static_cast<double>(duration_cast<microseconds>(elapsed).count()) / TESTS);
    float c = 0.f;
    for (auto i = 0ll; i < TESTS; ++i) {
        c += mathsimd::float3::dot(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    std::cout<<"Value of sum is "<<c<<std::endl;
}

void mathtests::benchmark_seq_cross() {
    auto& vectors = generate_seq_vectors();
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (auto i = 0ll; i < TESTS; ++i) {
        vectors[i % VALUES] = mathseq::float3::cross(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    auto elapsed = high_resolution_clock::now() - start;
    printf("Cross SEQ: %f\n", static_cast<double>(duration_cast<microseconds>(elapsed).count()) / TESTS);
    float c = 0.f;
    for (auto i = 0ll; i < TESTS; ++i) {
        c += mathseq::float3::dot(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    std::cout<<"Value of sum is "<<c<<std::endl;
}
