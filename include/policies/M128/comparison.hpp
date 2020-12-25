#ifndef MATHEMATICS_COMPARISON_HPP
#define MATHEMATICS_COMPARISON_HPP

#include "../M128.hpp"

template<typename T> template<typename TBool>
struct mathsimd::M128<T>::Comparison
{
	static constexpr size_t bits_per_byte = 8;
	static_assert(std::conjunction<
		std::is_convertible<TBool, char*>,
		std::negation<std::is_pointer<TBool>>,
		std::bool_constant<(sizeof(TBool) * bits_per_byte) >= float_count>>::value);
private:
	template<size_t RegisterIdx>
	static constexpr size_t bit_offset()
	{
		return (RegisterIdx % (bits_per_byte / aligned_floats)) * aligned_floats;
	}

	template<size_t RegisterIdx>
	static constexpr char clear_mask()
	{
		return ((1u << aligned_floats) - 1) << (aligned_floats * !bit_offset<RegisterIdx>());
	}

	template<size_t RegisterIdx> //todo deal with odd aligned float counts
	static void set(TBool& bitfield, int const &result)
	{
		auto* ptr = static_cast<char*>(bitfield) + RegisterIdx * aligned_floats / bits_per_byte;
		char const mask = (result << bit_offset<RegisterIdx>());

		*ptr &= clear_mask<RegisterIdx>();
		*ptr |= mask;
	}
#define COMPARISON_OP(FUNC,OP) \
private: \
	template<size_t N> \
    static void FUNC(TBool& result, T const& left, T const& right) \
    { \
        set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(load<N>(left), load<N>(right)))); \
    } \
    template<size_t N> \
    static void FUNC(TBool& result, T const& left, Register const& right) \
    { \
        set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(load<N>(left), right))); \
    } \
    template<size_t N> \
    static void FUNC(TBool& result, Register const& left, T const& right) \
    { \
        set<N>(result, _mm_movemask_ps(_mm_##OP##_ps(left, load<N>(right)))); \
    } \
    template<size_t... Idx> \
    static TBool FUNC(T const& left, T const& right, std::index_sequence<Idx...>) \
    { \
        TBool result; \
        (FUNC<Idx>(result, left, right), ...); \
        return result; \
    } \
    template<size_t... Idx> \
    static TBool FUNC(float const& left, T const& right, std::index_sequence<Idx...>) \
    { \
        TBool result; \
        (FUNC<Idx>(result, broadcast(left), right), ...); \
        return result; \
    } \
    template<size_t... Idx> \
    static TBool FUNC(T const& left, float const& right, std::index_sequence<Idx...>) \
    { \
        TBool result; \
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
