#ifndef MATHEMATICS_UTILITY_HPP
#define MATHEMATICS_UTILITY_HPP

#include <type_traits>
#include <utility>
#include <tuple>
namespace mathsimd::utility
{
	template<typename TSrc, typename TDst>
	struct _copy_cv { using type = TDst; };
	template<typename TSrc, typename TDst>
	struct _copy_cv<TSrc const, TDst> { using type = TDst const; };
	template<typename TSrc, typename TDst>
	struct _copy_cv<TSrc volatile, TDst> { using type = TDst volatile; };
	template<typename TSrc, typename TDst>
	struct _copy_cv<TSrc volatile const, TDst> { using type = TDst volatile const; };

	template<typename TSrc, typename TDst>
	struct copy_cv : _copy_cv<typename std::remove_reference<TSrc>::type, std::decay_t<TDst>>
	{
	};

	template<typename TSrc, typename TDst>
	using copy_cv_t = typename copy_cv<TSrc, TDst>::type;

	template<typename TOut, typename T>
	constexpr decltype(auto) smart_cast(T&& obj)
	{
		if constexpr (std::is_pointer<TOut>::value && !std::is_pointer<std::decay_t<T>>::value)
		{
			return static_cast<copy_cv_t<T, TOut>*>(std::forward<T>(obj));
		}
		else
		{
			return static_cast<copy_cv_t<T, TOut>>(std::forward<T>(obj));
		}
	}

	static size_t constexpr ceil_div(size_t dividend, size_t divisor) { return 1 + ((dividend - 1) / divisor); }
	static size_t constexpr not_zero(size_t n) { return n ? n : 1; }
	static bool constexpr is_pow2(size_t value) { return !(value & (value - 1)); }
	template<typename T>
	static size_t constexpr size(size_t count) { return count * sizeof(T); }
	template<typename T>
	static size_t constexpr count(size_t size) { return ceil_div(size, sizeof(T)); }

}

#endif //MATHEMATICS_UTILITY_HPP
