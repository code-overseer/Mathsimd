#ifndef MATHEMATICS_COMPARISON_HPP
#define MATHEMATICS_COMPARISON_HPP

#include "../M128.hpp"

template<typename T> template<typename TBitfield>
struct mathsimd::M128<T>::Comparison
{
	static constexpr size_t bits_per_byte = 8;
	static constexpr size_t bit_alignment = TBitfield::bit_alignment();
	static constexpr size_t bit_count = TBitfield::size();
	static_assert(std::conjunction<
		std::is_convertible<TBitfield, char*>,
		std::bool_constant<bit_count >= float_count>,
		std::bool_constant<bit_alignment == Base::template alignment<float>()>>::value); // bitfield must have inner bit alignment

	template<size_t RegisterIdx>
	static constexpr size_t offset()
	{
		return (RegisterIdx % (bits_per_byte / bit_alignment));
	}

	template<size_t RegisterIdx>
	static constexpr char clear_mask()
	{
		return ((1u << bit_alignment) - 1) << offset<RegisterIdx>();
	}

	template<size_t RegisterIdx>
	static void set(TBitfield& bitfield, int const &result)
	{
		auto* ptr = static_cast<char*>(bitfield) + RegisterIdx * bit_alignment / bits_per_byte;
		char const mask = (result << (offset<RegisterIdx>() * bit_alignment));

		*ptr &= clear_mask<RegisterIdx>();
		*ptr |= mask;
	}

#define HELPER_FUNC(FUNC,OP) \
	template<size_t N> \
    static void FUNC(TBitfield& result, T const& left, T const& right) \
    { \
        set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(load<N>(left), load<N>(right)))); \
    } \
    template<size_t N> \
    static void FUNC(TBitfield& result, T const& left, Register const& right) \
    { \
        set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(load<N>(left), right))); \
    } \
    template<size_t N> \
    static void FUNC(TBitfield& result, Register const& left, T const& right) \
    { \
        set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(left, load<N>(right)))); \
    } \
    template<size_t... Idx> \
    static TBitfield FUNC(T const& left, T const& right, std::index_sequence<Idx...>) \
    { \
        TBitfield result; \
        (FUNC<Idx>(result, left, right), ...); \
        return result; \
    } \
    template<size_t... Idx> \
    static TBitfield FUNC(float const& left, T const& right, std::index_sequence<Idx...>) \
    { \
        TBitfield result; \
        (FUNC<Idx>(result, broadcast(left), right), ...); \
        return result; \
    } \
    template<size_t... Idx> \
    static TBitfield FUNC(T const& left, float const& right, std::index_sequence<Idx...>) \
    { \
        TBitfield result; \
        (FUNC<Idx>(result, left, broadcast(right)), ...); \
        return result; \
    }

	HELPER_FUNC(less,cmplt)
	HELPER_FUNC(greater,cmpgt)
	HELPER_FUNC(greater_equals,cmpge)
	HELPER_FUNC(less_equals,cmple)
	HELPER_FUNC(equals,cmpeq)
	HELPER_FUNC(not_equals,cmpneq)

#undef HELPER_FUNC
};

#define COMPARISON_OPS(FUNC) \
template<typename T> template<typename TBitField> \
decltype(auto) mathsimd::M128<T>::FUNC(T const& left, T const& right) \
{ \
	return Comparison<TBitField>::FUNC(left, right, std::make_index_sequence<register_count>{}); \
} \
template<typename T> template<typename TBitField> \
decltype(auto) mathsimd::M128<T>::FUNC(T const& left, float const& right) \
{ \
	return Comparison<TBitField>::FUNC(left, right, std::make_index_sequence<register_count>{}); \
} \
template<typename T> template<typename TBitField> \
decltype(auto) mathsimd::M128<T>::FUNC(float const& left, T const& right) \
{ \
	return Comparison<TBitField>::FUNC(left, right, std::make_index_sequence<register_count>{}); \
}

COMPARISON_OPS(less)
COMPARISON_OPS(greater)
COMPARISON_OPS(less_equals)
COMPARISON_OPS(greater_equals)
COMPARISON_OPS(not_equals)
COMPARISON_OPS(equals)
#undef COMPARISON_OPS

#endif //MATHEMATICS_COMPARISON_HPP
