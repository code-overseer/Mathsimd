#include "tests.hpp"
#include "../include/constants.hpp"


constexpr int SEED = 1234;

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
