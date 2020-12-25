#include "tests.hpp"
#include "../include/mathsimd.hpp"
#include <cmath>
#include <cassert>
#include <ctime>
#include <utility>

constexpr unsigned int TESTS = 1000000u;
constexpr unsigned int VALUES = 100u;
constexpr int SEED = 1234;

template<typename T1, typename T2>
static void assert_exact(T1 const& normal, T2 const& simd)
{
    static_assert(T1::length == T2::length);
    auto n = static_cast<float const*>(normal);
    auto s = static_cast<float const*>(simd);
    for (auto i = 0; i < T1::length; ++i)
    {
        assert(n[i] == s[i]);
    }
}

static bool equal(float a, float b, float epsilon) {

	const float absA = std::fabs(a);
	const float absB = std::fabs(b);
	const float diff = std::fabs(a - b);

	if (a == b)
	{
		return true;
	}
	else if (a == 0 || b == 0 || diff < mathsimd::MIN_F)
	{
		return diff < (epsilon * mathsimd::MIN_F);
	}
	else
	{
		return diff / (absA + absB) < epsilon;
	}
}

static void assert_equals(float const& normal, float const& simd, float epsilon = mathsimd::EPSILON_F)
{
	if (!equal(normal, simd, epsilon))
	{
		printf("%f %f\n", normal, simd);
		throw -1;
	}
}

template<typename T1, typename T2>
static void assert_equals(T1 const& normal, T2 const& simd, float epsilon = mathsimd::EPSILON_F)
{
    static_assert(T1::length == T2::length);
    auto n = static_cast<float const*>(normal);
    auto s = static_cast<float const*>(simd);
    for (auto i = 0; i < T1::length; ++i)
    {
        assert_equals(n[i],s[i], epsilon);
    }
}

template<size_t N>
static void assert_equals(mathsimd::Bool<N> const& normal, mathsimd::Bool<N> const& simd)
{
	assert(normal == simd);
}

template <typename T>
struct TypeName { static const char *value; };
template<template<typename> typename Vector, typename Actual, typename Expected>
struct Runner
{
private:

	template<typename T>
	static float rnd()
	{
		static int seed = SEED;
		seed = int(std::fmod(static_cast<float>(seed) * 1373.f + 691.f, 509.f));
		if (seed)
		{
			return static_cast<float>(seed) / 509.f;
		}
		return rnd<T>();
	}

	template <typename P, size_t... Is>
	static Vector<P> Create(std::index_sequence<Is...>)
	{
		const auto f = [] (size_t){ return (rnd<P>() + rnd<P>()); };
		return Vector<P>{f(std::integral_constant<size_t, Is>{})...};
	}

	template<typename P>
	static Vector<P> Create()
	{
		return Create<P>(std::make_index_sequence<Vector<P>::length>{});
	}


public:
#define UNARY_EQUALS(FUNC, VEC, P1, P2, ERROR) \
	puts(#FUNC); \
	assert_equals(FUNC(Create<P1>()), FUNC(Create<P2>()), ERROR);

#define BINARY_EQUALS(FUNC, VEC, P1, P2, ERROR) \
	puts(#FUNC); \
	assert_equals(FUNC(Create<P1>(),Create<P1>()), FUNC(Create<P2>(), Create<P2>()), ERROR);

#define UNARY_EXACT(FUNC, VEC, P1, P2) \
	puts(#FUNC); \
	assert_exact(FUNC(Create<P1>()), FUNC(Create<P2>()));

#define BINARY_EXACT(FUNC, VEC, P1, P2) \
	puts(#FUNC); \
	assert_exact(FUNC(Create<P1>(),Create<P1>()), FUNC(Create<P2>(), Create<P2>()));

#define BINARY_OPS_EQUALS(OP, VEC, P1, P2, ERROR) \
	printf("operator %s\n", #OP); \
	assert_equals((Create<P1>() OP Create<P1>()), (Create<P2>() OP Create<P2>()), ERROR);

#define COMPARISONS(OP, VEC, P1, P2) \
	printf("operator %s\n", #OP); \
	assert_equals((Create<P1>() OP Create<P1>()), (Create<P2>() OP Create<P2>()));

#define RUN_TEST(VEC, ACTUAL, EXPECTED) \
	UNARY_EQUALS(reciprocal, VEC, ACTUAL, EXPECTED, mathsimd::FAST_ERROR_F); \
	UNARY_EXACT(absolute, VEC, ACTUAL, EXPECTED); \
	UNARY_EXACT(sign, VEC, ACTUAL, EXPECTED); \
	BINARY_EXACT(maximum, VEC, ACTUAL, EXPECTED); \
	BINARY_EXACT(minimum, VEC, ACTUAL, EXPECTED); \
	BINARY_OPS_EQUALS(+, VEC, ACTUAL, EXPECTED, mathsimd::EPSILON_F) \
	BINARY_OPS_EQUALS(-, VEC, ACTUAL, EXPECTED, mathsimd::EPSILON_F) \
	BINARY_OPS_EQUALS(/, VEC, ACTUAL, EXPECTED, mathsimd::EPSILON_F) \
	BINARY_OPS_EQUALS(*, VEC, ACTUAL, EXPECTED, mathsimd::EPSILON_F) \
	COMPARISONS(<, VEC, ACTUAL, EXPECTED) \
	COMPARISONS(>, VEC, ACTUAL, EXPECTED) \
	COMPARISONS(<=, VEC, ACTUAL, EXPECTED) \
	COMPARISONS(>=, VEC, ACTUAL, EXPECTED) \
	COMPARISONS(==, VEC, ACTUAL, EXPECTED) \
	COMPARISONS(!=, VEC, ACTUAL, EXPECTED)
//	BINARY_EQUALS(dot, VEC, ACTUAL, EXPECTED, mathsimd::EPSILON_F); \
//	UNARY_EQUALS(magnitude, VEC, ACTUAL, EXPECTED, mathsimd::FAST_ERROR_F); \
//	UNARY_EQUALS(normalize, VEC, ACTUAL, EXPECTED, mathsimd::FAST_ERROR_F); \
//	UNARY_EQUALS(sqr_magnitude, VEC, ACTUAL, EXPECTED, mathsimd::EPSILON_F); \

	template<size_t N>
 	static void Run(char const (&name)[N])
	{
		printf("%s Tests Begin\n", name);
 		RUN_TEST(Vector, Actual, Expected);
		printf("%s Tests End\n\n", name);
	}
};

void mathtests::RunTests()
{
	using namespace mathsimd;
	Runner<mathsimd::Float2, mathsimd::SimdVectorPolicy<float>::M128, mathsimd::CMathPolicy<float>::Vector>::Run("Float2");
	Runner<mathsimd::Float3, mathsimd::SimdVectorPolicy<float>::M128, mathsimd::CMathPolicy<float>::Vector>::Run("Float3");
	Runner<mathsimd::Float4, mathsimd::SimdVectorPolicy<float>::M128, mathsimd::CMathPolicy<float>::Vector>::Run("Float4");
}

// void mathtests::test_float4_dot() {
//     using namespace mathsimd;
//     float ref_a[4], ref_b[4];
//     for (auto &i: ref_a) { i = rnd(); }
//     for (auto &i: ref_b) { i = rnd(); }
//     float4 a(ref_a[0],ref_a[1], ref_a[2],ref_a[3]),b(ref_b[0],ref_b[1],ref_b[2],ref_b[3]);
//     auto actual = dot(a,b);
//     auto expected = std::inner_product(ref_a, ref_a+4, ref_b, 0.f);
//     assert(std::fabs(actual - expected) < EPSILON_F);
// }

// void mathtests::test_float4_cross() {
//     using namespace mathsimd;
//     float ref_a[4], ref_b[4];
//     for (auto &i: ref_a) { i = rnd(); }
//     for (auto &i: ref_b) { i = rnd(); }
//     float4 a(ref_a[0],ref_a[1], ref_a[2],ref_a[3]),b(ref_b[0],ref_b[1],ref_b[2],ref_b[3]);
//     float4 expected(ref_a[1]*ref_b[2] - ref_a[2]*ref_b[1],
//                     ref_a[2]*ref_b[0] - ref_a[0]*ref_b[2],
//                     ref_a[0]*ref_b[1] - ref_a[1]*ref_b[0],
//                     ref_a[3]*ref_b[3] - ref_a[3]*ref_b[3]);
//     auto actual = cross(a, b);
//     assert((actual == expected).all_true());
// }

// void mathtests::test_float3_dot() {
//     using namespace mathsimd;
//     using nfloat3 = float3<OperationType::Normal>;
//     float ref_a[3], ref_b[3];
//     for (auto &i: ref_a) { i = rnd(); }
//     for (auto &i: ref_b) { i = rnd(); }
//     float3 ac_a(ref_a[0],ref_a[1], ref_a[2]),ac_b(ref_b[0],ref_b[1],ref_b[2]);
//     nfloat3 ex_a(ref_a[0],ref_a[1], ref_a[2]),ex_b(ref_b[0],ref_b[1],ref_b[2]);

//     auto expected = dot(ex_a,ex_b);
//     auto actual = dot(ac_a,ac_b);
//     assert(std::fabs(actual - expected) < EPSILON_F);
// }

// void mathtests::test_float3_cross() {
//     using namespace mathsimd;
//     float ref_a[3], ref_b[3];
//     for (auto &i: ref_a) { i = rnd(); }
//     for (auto &i: ref_b) { i = rnd(); }
//     float3 a(ref_a[0],ref_a[1], ref_a[2]),b(ref_b[0],ref_b[1],ref_b[2]);
//     float3 expected(ref_a[1]*ref_b[2] - ref_a[2]*ref_b[1],
//                     ref_a[2]*ref_b[0] - ref_a[0]*ref_b[2],
//                     ref_a[0]*ref_b[1] - ref_a[1]*ref_b[0]);
//     auto actual = cross(a, b);
//     assert((actual == expected).all_true());
// }

// using M44 = std::array<std::array<float,4>,4>;
// static M44 randmat()
// {
//     M44 M;
//     for (int i=0; i < 4; i++) {
//         for (int j=0; j < 4; j++) {
//             M[i][j] = rnd();
//         }
//     }
//     return M;
// }

// static mathsimd::float4x4 copy(M44 const &a)
// {
//     __m128 t[]{_mm_loadu_ps(a[0].data()),
//                _mm_loadu_ps(a[1].data()),
//                _mm_loadu_ps(a[2].data()),
//                _mm_loadu_ps(a[3].data())};

//     return mathsimd::float4x4(t[0],t[1],t[2],t[3]);
// }

// static M44 operator*(M44 const &a, M44 const &b) {
//     M44 t;
//     for (volatile int i=0; i < 4; i++)
//         for (volatile int j=0; j < 4; j++)
//             t[j][i] = a[0][i]*b[j][0] + a[1][i]*b[j][1] + a[2][i]*b[j][2] + a[3][i]*b[j][3];
//     return t;
// }

// void mathtests::test_float4x4_matmul() {
//     using namespace mathsimd;
//     M44 A = randmat();
//     M44 B = randmat();
//     float4x4 a = copy(A);
//     float4x4 b = copy(B);
//     float4x4 out = matmul(a,b);
//     M44 ref = A*B;

//     assert(!memcmp(static_cast<float const *>(out), &ref[0], sizeof(out)));
// }

// void mathtests::test_float4x4_vecmul() {
//     using namespace mathsimd;
//     M44 A = randmat();
//     float ref_b[4];
//     for (auto &i: ref_b) { i = rnd(); }
//     float4x4 a = copy(A);
//     float4 b(ref_b[0],ref_b[1],ref_b[2], ref_b[3]);
//     float4 out = matmul(a,b);
//     float ref_out[4];
//     for (int i=0; i < 4; i++)
//             ref_out[i] = A[0][i]*ref_b[0] + A[1][i]*ref_b[1] + A[2][i]*ref_b[2] + A[3][i]*ref_b[3];

//     auto tmp = float4(_mm_loadu_ps(ref_out));

//     assert((tmp == out).all_true());
// }

