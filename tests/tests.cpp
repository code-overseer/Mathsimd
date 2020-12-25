#include <cstdio>
#include "../include/constants.hpp"


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

static bool equals(float a, float b, float epsilon)
{
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
		return diff / (std::fabs(a) + std::fabs(b)) < epsilon;
	}
}

static void assert_equals(float const& normal, float const& simd, float epsilon)
{
	if (!equals(normal, simd, epsilon))
	{
		printf("%f %f\n", normal, simd);
		throw -1;
	}
}

template<typename T1, typename T2>
static void assert_equals(T1 const& normal, T2 const& simd, float epsilon)
{
	static_assert(T1::length == T2::length);
	auto n = static_cast<float const*>(normal);
	auto s = static_cast<float const*>(simd);
	for (auto i = 0; i < T1::length; ++i)
	{
		assert_equals(n[i],s[i], epsilon);
	}
}

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
