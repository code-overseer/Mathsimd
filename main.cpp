#include "include/policies/M128.hpp"
#include <cstdio>
//#include "tests/tests.hpp"

struct float4x4
{
	static constexpr size_t size() { return 16; }
	static constexpr size_t aligned_floats() { return 8; }
	static constexpr size_t alignment() { return 32; }
	alignas(32) float values[4][4]{0.0f};
	inline operator float*() noexcept { return *values; }
	inline operator float const*() const noexcept { return *values; }
	inline float& operator[](size_t const idx) { return values[idx / 4][idx % 4]; }
	inline float operator[](size_t const idx) const { return values[idx / 4][idx % 4]; }
	void print() const
	{
		for (auto i = 0; i < 4; ++i)
		{
			printf("(%f ", values[0][i]);
			for (auto j = 1; j < 4; ++j)
			{
				printf(", %f", values[j][i]);
			}
			puts(")");
		}
	}
};

int main()
{
	using namespace mathsimd;
	float4x4 f;
	f.print();

    return 0;
}
