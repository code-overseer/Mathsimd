#ifndef MATHEMATICS_M128_HPP
#define MATHEMATICS_M128_HPP
#include <immintrin.h>
#include <type_traits>
#include <utility>
#include "../utility.hpp"
#include "register_policy.hpp"

namespace mathsimd
{
	template<typename T>
	struct M128 : RegisterPolicy<T, __m128>
	{
	private:
		using Base = RegisterPolicy<T, __m128>;
		using Register = __m128;
		using Base::alignment;
		using Base::active_size;
	public:
		static size_t constexpr float_count = Base::template active_size<float>();
		static size_t constexpr aligned_floats = Base::template active_aligned<float>();
		static size_t constexpr floats_per_register = Base::template register_size<float>();
		static size_t constexpr register_count = (float_count / aligned_floats);
		static_assert(std::conjunction<std::is_convertible<T, float const*>, // defined T::operator float const*()
		    std::bool_constant<Base::is_pow2(Base::alignment())>, // alignment must be power of 2 and greater than 8
			std::bool_constant<Base::is_pow2(aligned_floats)>, // cannot load 3 unaligned floats into register
			std::bool_constant<(register_count > 0)>, //  at least 1 register
			std::bool_constant<(Base::alignment() <= alignof(Register) || aligned_floats == 4)>>::value); // prevents alternating aligned floats

		static Register load(float const& value)
		{
			return _mm_load_ss(&value);
		}

		static void store(float& dst, Register const& src)
		{
			_mm_store_ss(&dst, src);
		}

		template<size_t N = 0>
		static Register load(T const& vector)
		{
			static size_t constexpr offset = N * Base::template alignment<float>(Base::alignment);
			// From Base, active_aligned will always be < 128 bits
			if constexpr (Base::is_aligned())
			{
				return _mm_load_ps(cast(vector) + offset);
			}
			else if constexpr (aligned_floats == 1)
			{
				return load(*(cast(vector) + offset));
			}
			else if constexpr (aligned_floats == 2)
			{
				return _mm_castsi128_ps(_mm_loadu_si64(cast(vector) + offset));
			}
			else
			{
				return _mm_loadu_ps(cast(vector) + offset);
			}
		}

		template<size_t N = 0>
		static void store(T& dst, Register const& src)
		{
			static size_t constexpr offset = N * Base::template alignment<float>(Base::alignment);
			if constexpr (Base::is_aligned())
			{
				_mm_store_ps(cast(dst) + offset, src);
			}
			else if constexpr (aligned_floats == 1)
			{
				store(*(cast(dst) + offset), src);
			}
			else if constexpr (aligned_floats == 2)
			{
				_mm_storeu_si64(cast(dst) + offset, _mm_castps_si128(src));
			}
			else // float2 special-case
			{
				_mm_storeu_ps(cast(dst) + offset, src);
			}
		}

		static Register broadcast(float const& value)
		{
			return _mm_broadcast_ss(&value);
		}

		struct Binary;
		template<typename BitField>
		struct Comparison;
		struct Unary;
		struct Recursion;
	};
}

#include "M128/binary.hpp"
#include "M128/comparison.hpp"
#include "M128/recursion.hpp"
#include "M128/unary.hpp"

#endif //MATHEMATICS_M128_HPP
