#ifndef MATHEMATICS_REGISTER_IO_HPP
#define MATHEMATICS_REGISTER_IO_HPP

#include "../M128.hpp"

/// Unused, WIP
template<typename T> template<size_t N, bool IsAligned, bool IsExcess>
struct mathsimd::M128<T>::RegisterIO
{
public:
	static Register load(T const& vector, [[maybe_unused]] float const &init = 0.0f)
	{
		if constexpr(IsAligned)
		{
			return _mm_load_ps(cast(vector) + N * sizeof(Register) / sizeof(float));
		}
		else
		{
			return _mm_loadu_ps(cast(vector) + N * sizeof(Register) / sizeof(float));
		}
	}

	static void store(T& dst, Register const& src)
	{
		if constexpr(IsAligned)
		{
			_mm_store_ps(cast(dst) + N * floats_per_register, src);
		}
		else
		{
			_mm_storeu_ps(cast(dst) + N * floats_per_register, src);
		}
	}
};

template<typename T> template<bool IsAligned>
struct mathsimd::M128<T>::RegisterIO<mathsimd::M128<T>::register_count - 1, IsAligned, true>
{
private:
	template<size_t Excess = float_count % floats_per_register>
	static Register blender();

	template<>
	static Register blender<1>()
	{
		float vals[]{0.0f,-0.0f,-0.0f,-0.0f};
		return _mm_load_ps(vals);
	}

	template<>
	static Register blender<2>()
	{
		float vals[]{-0.0f,-0.0f,0.0f,0.0f};
		return _mm_load_ps(vals);
	}

	template<>
	static Register blender<3>()
	{
		float vals[]{0.0f,0.0f,0.0f,-0.0f};
		return _mm_load_ps(vals);
	}

	template<size_t Excess = float_count % floats_per_register>
	static void excess_store(T& dst, Register const& src);

	template<>
	static void excess_store<1>(T& dst, Register const& src)
	{
		_mm_store_ss(cast(dst) + N * floats_per_register, src);
	}

	template<>
	static void excess_store<2>(T& dst, Register const& src)
	{
		_mm_storeu_si64(cast(dst) + N * floats_per_register, _mm_castps_si128(src));
	}

	template<>
	static void excess_store<3>(T& dst, Register const& src)
	{
		_mm_storeu_si64(cast(dst) + N * floats_per_register, _mm_castps_si128(src));
		_mm_store_ss(cast(dst) + N * floats_per_register + 2, _mm_shuffle_ps(src,src,_MM_SHUFFLE(3,3,3,3)));
	}

	static constexpr size_t N = mathsimd::M128<T>::register_count - 1;
public:
	static Register load(T const& vector, [[maybe_unused]] float const &init = 0.0f)
	{
		__m128 val;
		if constexpr(IsAligned)
		{
			val = _mm_load_ps(cast(vector) + N * sizeof(Register) / sizeof(float));
		}
		else
		{
			val = _mm_loadu_ps(cast(vector) + N * sizeof(Register) / sizeof(float));
		}
		return _mm_blend_ps(val, broadcast(init), blender());
	}

	static void store(T& dst, Register const& src)
	{
		if constexpr(IsAligned)
		{
			_mm_store_ps(cast(dst) + N * floats_per_register, src);
		}
		else
		{
			excess_store(dst,src);
		}
	}
};
#endif //MATHEMATICS_REGISTER_IO_HPP
