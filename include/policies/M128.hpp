#ifndef MATHEMATICS_M128_HPP
#define MATHEMATICS_M128_HPP
#include <immintrin.h>
#include <type_traits>
#include <utility>
#include "../utility.hpp"

namespace mathsimd
{
	struct M128
	{
		using Register = __m128;
	private:
		template<typename T>
		using valid = std::conjunction<
				std::is_convertible<typename std::decay<T>::type, float*>,
				std::negation<std::is_pointer<typename std::decay<T>::type>>>;

		template<typename T>
		static constexpr size_t float_count() { return (sizeof(typename std::decay<T>::type) / sizeof(float)); }
		static constexpr size_t BITS_PER_BYTE = 8;

		template<typename BitField, typename T>
		struct bitfield_check : std::conjunction<
				std::is_convertible<BitField, char*>,
				std::negation<std::is_pointer<BitField>>,
				std::bool_constant<(sizeof(BitField) * BITS_PER_BYTE) >= (float_count<T>())>>
		{
		};
		template<typename T>
		static constexpr auto cast(T&& vector) { return smart_cast<float*>(vector); }
		template<typename T>
		static constexpr size_t chunks() { return sizeof(typename std::decay<T>::type) / sizeof(Register); }

		template<size_t ChunkIdx, typename BitField>
		static void set(BitField& bitfield, int const &result)
		{
			auto* ptr = static_cast<char*>(bitfield) + ChunkIdx * float_count<Register>() / BITS_PER_BYTE;
			static constexpr size_t offset = ChunkIdx % (BITS_PER_BYTE / float_count<Register>());
			static constexpr char clear_mask = ((1 << float_count<Register>()) - 1) << (float_count<Register>() * !offset);
			char const mask = (result << (offset * float_count<Register>()));

			*ptr &= clear_mask;
			*ptr |= mask;
		}
	public:
		template<size_t N = 0, typename T>
		static Register load(T&& vector)
		{
			static_assert(valid<T>::value);
			if constexpr(!(alignof(typename std::decay<T>::type) % alignof(Register)))
			{
				return _mm_load_ps(cast(vector) + N * sizeof(Register) / sizeof(float));
			}
			else
			{
				return _mm_loadu_ps(cast(vector) + N * sizeof(Register) / sizeof(float));
			}
		}

		template<size_t N = 0, typename T>
		static void store(T& dst, Register const& src)
		{
			static_assert(valid<T>::value);
			if constexpr(!(alignof(T) % alignof(Register)))
			{
				_mm_store_ps(cast(dst) + N * sizeof(Register) / sizeof(float), src);
			}
			else
			{
				_mm_storeu_ps(cast(dst) + N * sizeof(Register) / sizeof(float), src);
			}
		}

	private:

#define PRIVATE_BINARY_OP(FUNC,OP) \
		template<size_t N, typename T> \
		static void FUNC(typename std::decay<T>::type& result, T&& left, T&& right) \
		{ \
			store<N>(result, _mm_##OP##_ps(load<N>(left), load<N>(right))); \
		} \
		template<typename T, size_t... Idx> \
		static typename std::decay<T>::type FUNC(T&& left, T&& right, std::index_sequence<Idx...>) \
		{ \
			typename std::decay<T>::type result; \
			(FUNC<Idx>(result, std::forward<T>(left), std::forward<T>(right)),...); \
			return result; \
		}

#define PRIVATE_BINARY_WITH_SCALAR(FUNC,OP) \
		template<size_t N, typename T> \
		static void FUNC(typename std::decay<T>::type& result, T&& left, Register const& right) \
		{ \
			store<N>(result, _mm_##OP##_ps(load<N>(left), right)); \
		} \
		template<size_t N, typename T> \
		static void FUNC(typename std::decay<T>::type& result, Register const& left, T&& right) \
		{ \
			store<N>(result, _mm_##OP##_ps(left, load<N>(right))); \
		} \
		template<typename T, size_t... Idx> \
		static typename std::decay<T>::type FUNC(T&& left, float const& right, std::index_sequence<Idx...>) \
		{ \
			typename std::decay<T>::type result; \
			auto const r = _mm_broadcast_ss(&right); \
			(FUNC<Idx>(result, std::forward<T>(left), r),...); \
			return result; \
		} \
		template<typename T, size_t... Idx> \
		static typename std::decay<T>::type FUNC(float const& left, T&& right, std::index_sequence<Idx...>) \
		{ \
			typename std::decay<T>::type result; \
			auto const l = _mm_broadcast_ss(&left); \
			(FUNC<Idx>(result, l, std::forward<T>(right)),...); \
			return result; \
		}

		PRIVATE_BINARY_OP(add, add)
		PRIVATE_BINARY_OP(subtract, sub)
		PRIVATE_BINARY_OP(multiply, mul)
		PRIVATE_BINARY_OP(divide, div)
		PRIVATE_BINARY_OP(minimum, min)
		PRIVATE_BINARY_OP(maximum, max)

#undef PRIVATE_BINARY_OP
#undef PRIVATE_BINARY_WITH_SCALAR

		template<size_t N, typename T>
		static void reciprocal(typename std::decay<T>::type& result, T&& vector)
		{
			store<N>(result, _mm_rcp_ps(load<N>(vector)));
		}

		template<size_t N, typename T>
		static void sign(typename std::decay<T>::type& result, T&& vector)
		{
			static constexpr float plus = 1.0f;
			static constexpr float minus = -1.0f;
			auto reg = load<N>(vector);
			auto zero = _mm_setzero_ps();
			auto positive = _mm_and_ps(_mm_cmpgt_ps(reg, zero), _mm_broadcast_ss(&plus));
			auto negative = _mm_and_ps(_mm_cmplt_ps(reg, zero), _mm_broadcast_ss(&minus));
			store<N>(result, _mm_or_ps(positive, negative));
		}

		template<size_t N, typename T>
		static void absolute(typename std::decay<T>::type& result, T&& vector)
		{
			auto mask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
			store<N>(result, _mm_and_ps(load<N>(vector), mask));
		}

#define PRIVATE_UNARY_FORWARD(FUNC) \
		template<typename T, size_t... Idx> \
		static typename std::decay<T>::type FUNC(T&& vector, std::index_sequence<Idx...>) \
		{ \
			typename std::decay<T>::type result; \
			(FUNC<Idx>(result, std::forward<T>(vector)),...); \
			return result; \
		}

		PRIVATE_UNARY_FORWARD(reciprocal)
		PRIVATE_UNARY_FORWARD(sign)
		PRIVATE_UNARY_FORWARD(absolute)
#undef PRIVATE_UNARY_FORWARD

#define PRIVATE_CMP_OP(FUNC,OP) \
		template<size_t N, typename BitField, typename T> \
		static void FUNC(BitField& result, T&& left, T&& right) \
		{ \
			static_assert(bitfield_check<BitField,T>::value); \
			set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(load<N>(left), load<N>(right)))); \
		} \
		template<size_t N, typename BitField, typename T> \
		static void FUNC(BitField& result, T&& left, Register const& right) \
		{ \
			static_assert(bitfield_check<BitField,T>::value); \
			set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(load<N>(left), right))); \
		} \
		template<size_t N, typename BitField, typename T> \
		static void FUNC(BitField& result, Register const& left, T&& right) \
		{ \
			static_assert(bitfield_check<BitField,T>::value); \
			set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(left, load<N>(right)))); \
		} \
		template<typename BitField, typename T, size_t... Idx> \
		static BitField FUNC(T&& left, T&& right, std::index_sequence<Idx...>) \
		{ \
			BitField result; \
			(FUNC<Idx>(result, std::forward<T>(left), std::forward<T>(right)), ...); \
			return result; \
		} \
		template<typename BitField, typename T, size_t... Idx> \
		static BitField FUNC(float const& left, T&& right, std::index_sequence<Idx...>) \
		{ \
			BitField result; \
			auto const l = _mm_broadcast_ss(&left); \
			(FUNC<Idx>(result, l, std::forward<T>(right)), ...); \
			return result; \
		} \
		template<typename BitField, typename T, size_t... Idx> \
		static BitField FUNC(T&& left, float const& right, std::index_sequence<Idx...>) \
		{ \
			BitField result; \
			auto const r = _mm_broadcast_ss(&right); \
			(FUNC<Idx>(result, std::forward<T>(left), r), ...); \
			return result; \
		}

		PRIVATE_CMP_OP(less,cmplt)
		PRIVATE_CMP_OP(greater,cmpgt)
		PRIVATE_CMP_OP(greater_equals,cmpge)
		PRIVATE_CMP_OP(less_equals,cmple)
		PRIVATE_CMP_OP(equals,cmpeq)
		PRIVATE_CMP_OP(not_equals,cmpneq)

#undef PRIVATE_CMP_OP

	public:
#define PUBLIC_BINARY_WITH_SCALAR(FUNC) \
		template<typename T> \
		static decltype(auto) FUNC(T&& left, float const& right) \
		{ \
			return FUNC(std::forward<T>(left), right, std::make_index_sequence<chunks<T>()>{}); \
		} \
		template<typename T> \
		static decltype(auto) FUNC(float const& left, T&& right) \
		{ \
			return FUNC(left, std::forward<T>(right), std::make_index_sequence<chunks<T>()>{}); \
		}

#define PUBLIC_BINARY_OP(FUNC) \
		template<typename T> \
		static decltype(auto) FUNC(T&& left, T&& right) \
		{ \
			return FUNC(std::forward<T>(left), std::forward<T>(right), std::make_index_sequence<chunks<T>()>{}); \
		}

		PUBLIC_BINARY_OP(add)
		PUBLIC_BINARY_OP(subtract)
		PUBLIC_BINARY_OP(multiply)
		PUBLIC_BINARY_OP(divide)
		PUBLIC_BINARY_OP(minimum)
		PUBLIC_BINARY_OP(maximum)

		PUBLIC_BINARY_WITH_SCALAR(add)
		PUBLIC_BINARY_WITH_SCALAR(subtract)
		PUBLIC_BINARY_WITH_SCALAR(multiply)
		PUBLIC_BINARY_WITH_SCALAR(divide)
#undef PUBLIC_BINARY_OP
#undef PUBLIC_BINARY_WITH_SCALAR

#define PUBLIC_UNARY_OP(FUNC) \
    template<typename T> \
    static decltype(auto) FUNC(T&& vector) \
    { \
        return FUNC(std::forward<T>(vector), std::make_index_sequence<chunks<T>()>{}); \
    }

		PUBLIC_UNARY_OP(reciprocal)
		PUBLIC_UNARY_OP(sign)
		PUBLIC_UNARY_OP(absolute)
#undef PUBLIC_UNARY_OP

#define PUBLIC_CMP_OP(FUNC) \
		template<typename BitField, typename T> \
		static decltype(auto) FUNC(T&& left, T&& right) \
		{ \
			return FUNC<BitField>(std::forward<T>(left), std::forward<T>(right), std::make_index_sequence<chunks<T>()>{}); \
		} \
		template<typename BitField, typename T> \
		static decltype(auto) FUNC(T&& left, float const& right) \
		{ \
			return FUNC<BitField>(std::forward<T>(left), right, std::make_index_sequence<chunks<T>()>{}); \
		} \
		template<typename BitField, typename T> \
		static decltype(auto) FUNC(float const& left, T&& right) \
		{ \
			return FUNC<BitField>(left, std::forward<T>(right), std::make_index_sequence<chunks<T>()>{}); \
		}

		PUBLIC_CMP_OP(less)
		PUBLIC_CMP_OP(greater)
		PUBLIC_CMP_OP(greater_equals)
		PUBLIC_CMP_OP(less_equals)
		PUBLIC_CMP_OP(equals)
		PUBLIC_CMP_OP(not_equals)
#undef PUBLIC_CMP_OP

	};
}

#endif //MATHEMATICS_M128_HPP
