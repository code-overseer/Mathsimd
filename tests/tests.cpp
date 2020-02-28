#include "tests.hpp"
#include <numeric>
#include <iostream>
constexpr unsigned int TESTS = 1000000u;
constexpr unsigned int VALUES = 100u;
constexpr int SEED = 1234;

static float rnd() {
    static int seed = SEED;
    seed = int(std::fmod(static_cast<float>(seed) * 1373.f + 691.f, 509.f));
    return static_cast<float>(seed) / 509.f;
}

static unsigned int rnd_idx() {
    return static_cast<unsigned int>(std::time(nullptr)) % VALUES;
}

void mathtests::test_float2_dot() {
    using namespace mathsimd;
    float ref_a[2], ref_b[2];
    for (auto &i: ref_a) { i = rnd(); }
    for (auto &i: ref_b) { i = rnd(); }
    float2 a(ref_a[0],ref_a[1]),b(ref_b[0],ref_b[1]);
    auto actual = dot(a,b);
    auto expected = std::inner_product(ref_a, ref_a+2, ref_b, 0.f);
    std::cout.precision(20);
    int ac, e;
    memcpy(&ac, &actual, sizeof(ac));
    memcpy(&e, &expected, sizeof(e));
    assert(std::fabs(expected - actual) < EPSILON_F);
}

void mathtests::test_float4_dot() {
    using namespace mathsimd;
    float ref_a[4], ref_b[4];
    for (auto &i: ref_a) { i = rnd(); }
    for (auto &i: ref_b) { i = rnd(); }
    float4 a(ref_a[0],ref_a[1], ref_a[2],ref_a[3]),b(ref_b[0],ref_b[1],ref_b[2],ref_b[3]);
    auto actual = dot(a,b);
    auto expected = std::inner_product(ref_a, ref_a+4, ref_b, 0.f);
    assert(std::fabs(actual - expected) < EPSILON_F);
}

void mathtests::test_float4_cross() {
    using namespace mathsimd;
    float ref_a[4], ref_b[4];
    for (auto &i: ref_a) { i = rnd(); }
    for (auto &i: ref_b) { i = rnd(); }
    float4 a(ref_a[0],ref_a[1], ref_a[2],ref_a[3]),b(ref_b[0],ref_b[1],ref_b[2],ref_b[3]);
    float4 expected(ref_a[1]*ref_b[2] - ref_a[2]*ref_b[1],
                    ref_a[2]*ref_b[0] - ref_a[0]*ref_b[2],
                    ref_a[0]*ref_b[1] - ref_a[1]*ref_b[0],
                    ref_a[3]*ref_b[3] - ref_a[3]*ref_b[3]);
    auto actual = cross(a, b);
    assert(actual == expected);
}

void mathtests::test_float3_dot() {
    using namespace mathsimd;
    float ref_a[3], ref_b[3];
    for (auto &i: ref_a) { i = rnd(); }
    for (auto &i: ref_b) { i = rnd(); }
    float3 a(ref_a[0],ref_a[1], ref_a[2]),b(ref_b[0],ref_b[1],ref_b[2]);
    auto actual = dot(a,b);
    auto expected = std::inner_product(ref_a, ref_a+3, ref_b, 0.f);
    assert(std::fabs(actual - expected) < EPSILON_F);
}

void mathtests::test_float3_cross() {
    using namespace mathsimd;
    float ref_a[3], ref_b[3];
    for (auto &i: ref_a) { i = rnd(); }
    for (auto &i: ref_b) { i = rnd(); }
    float3 a(ref_a[0],ref_a[1], ref_a[2]),b(ref_b[0],ref_b[1],ref_b[2]);
    float3 expected(ref_a[1]*ref_b[2] - ref_a[2]*ref_b[1],
                    ref_a[2]*ref_b[0] - ref_a[0]*ref_b[2],
                    ref_a[0]*ref_b[1] - ref_a[1]*ref_b[0]);
    auto actual = cross(a, b);
    assert(actual == expected);
}

using M44 = std::array<std::array<float,4>,4>;
static M44 randmat()
{
    M44 M;
    for (int i=0; i < 4; i++) {
        for (int j=0; j < 4; j++) {
            M[i][j] = rnd();
        }
    }
    return M;
}

static mathsimd::float4x4 copy(M44 const &a)
{
    __m128 t[]{_mm_loadu_ps(a[0].data()),
               _mm_loadu_ps(a[1].data()),
               _mm_loadu_ps(a[2].data()),
               _mm_loadu_ps(a[3].data())};

    return mathsimd::float4x4(t[0],t[1],t[2],t[3]);
}

static M44 operator*(M44 const &a, M44 const &b) {
    M44 t;
    for (volatile int i=0; i < 4; i++)
        for (volatile int j=0; j < 4; j++)
            t[j][i] = a[0][i]*b[j][0] + a[1][i]*b[j][1] + a[2][i]*b[j][2] + a[3][i]*b[j][3];
    return t;
}

void mathtests::test_float4x4_matmul() {
    using namespace mathsimd;
    M44 A = randmat();
    M44 B = randmat();
    float4x4 a = copy(A);
    float4x4 b = copy(B);
    float4x4 out = matmul(a,b);
    M44 ref = A*B;

    assert(!memcmp(static_cast<float const *>(out), &ref[0], sizeof(out)));
}

void mathtests::test_float4x4_vecmul() {
    using namespace mathsimd;
    M44 A = randmat();
    float ref_b[4];
    for (auto &i: ref_b) { i = rnd(); }
    float4x4 a = copy(A);
    float4 b(ref_b[0],ref_b[1],ref_b[2], ref_b[3]);
    float4 out = matmul(a,b);
    float ref_out[4];
    for (int i=0; i < 4; i++)
            ref_out[i] = A[0][i]*ref_b[0] + A[1][i]*ref_b[1] + A[2][i]*ref_b[2] + A[3][i]*ref_b[3];

    auto tmp = float4(_mm_loadu_ps(ref_out));

    assert(tmp == out);
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

void mathtests::benchmark_simd_dot() {
    auto& vectors = generate_simd_vectors();
    using namespace std::chrono;
    float c = 0.f;
    auto start = high_resolution_clock::now();
    for (auto i = 0ll; i < TESTS; ++i) {
        c += mathsimd::dot(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    auto elapsed = high_resolution_clock::now() - start;
    printf("Dot SIMD: %f\n", static_cast<double>(duration_cast<microseconds>(elapsed).count()) / TESTS);
    std::cout<<"Value of sum is "<<c<<std::endl;
}

void mathtests::benchmark_simd_cross() {
    auto& vectors = generate_simd_vectors();
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (auto i = 0ll; i < TESTS; ++i) {
        vectors[i % VALUES] = mathsimd::cross(vectors[rnd_idx()],vectors[rnd_idx()]);

    }
    auto elapsed = high_resolution_clock::now() - start;
    printf("Cross SIMD: %f\n", static_cast<double>(duration_cast<microseconds>(elapsed).count()) / TESTS);
    float c = 0.f;
    for (auto i = 0ll; i < TESTS; ++i) {
        c += mathsimd::dot(vectors[rnd_idx()],vectors[rnd_idx()]);
    }
    std::cout<<"Value of sum is "<<c<<std::endl;
}

