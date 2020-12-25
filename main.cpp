#include "tests/tests.hpp"
#include "include/policies/M128.hpp"
struct float4x4
{
	static constexpr size_t length = 16;
	alignas(32) float vals[16]{0.0f};
	inline operator float*() noexcept { return vals; }
	inline operator float const*() const noexcept { return vals; }
	inline float& operator[](size_t const idx) { return vals[idx]; }
	inline float operator[](size_t const idx) const { return vals[idx]; }
	void print() const
	{
		for (auto i = 0; i < 4; ++i)
		{
			printf("(%f ", vals[i]);
			for (auto j = 1; j < 4; ++j)
			{
				printf(", %f", vals[i + j * 4]);
			}
			puts(")");
		}
	}
};

int main()
{
	using namespace mathsimd;
	M128<float4x4>::load(0.0f);
	mathtests::RunTests();

    return 0;
}
