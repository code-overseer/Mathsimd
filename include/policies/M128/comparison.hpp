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

private:
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
#define COMPARISON_OP(FUNC,OP) \
private: \
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
    } \
public: \
	static decltype(auto) FUNC(T const& left, T const& right) \
	{ \
		return FUNC(left, right, std::make_index_sequence<register_count>{}); \
	} \
	static decltype(auto) FUNC(T const& left, float const& right) \
	{ \
		return FUNC(left, right, std::make_index_sequence<register_count>{}); \
	} \
	static decltype(auto) FUNC(float const& left, T const& right) \
	{ \
		return FUNC(left, right, std::make_index_sequence<register_count>{}); \
	}

	COMPARISON_OP(less,cmplt)
	COMPARISON_OP(greater,cmpgt)
	COMPARISON_OP(greater_equals,cmpge)
	COMPARISON_OP(less_equals,cmple)
	COMPARISON_OP(equals,cmpeq)
	COMPARISON_OP(not_equals,cmpneq)

#undef COMPARISON_OP
};

#endif //MATHEMATICS_COMPARISON_HPP
