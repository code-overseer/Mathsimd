#ifndef MATHEMATICS_UTILITY_HPP
#define MATHEMATICS_UTILITY_HPP

#include <type_traits>
#include <utility>

template<typename TSrc, typename TDst>
struct _copy_cv { using type = TDst; };
template<typename TSrc, typename TDst>
struct _copy_cv<TSrc const, TDst> { using type = TDst const; };
template<typename TSrc, typename TDst>
struct _copy_cv<TSrc volatile, TDst> { using type = TDst volatile; };
template<typename TSrc, typename TDst>
struct _copy_cv<TSrc volatile const, TDst> { using type = TDst volatile const; };

template<typename TSrc, typename TDst>
struct copy_cv : _copy_cv<typename std::remove_reference<TSrc>::type, typename std::decay<TDst>::type>
{
};

template<typename TOut, typename T>
constexpr decltype(auto) smart_cast(T&& obj)
{
	if constexpr (std::is_pointer<TOut>::value && !std::is_pointer<typename std::decay<T>::type>::value)
	{
		return static_cast<typename copy_cv<T, typename std::remove_pointer<TOut>::type>::type*>(std::forward<T>(obj));
	}
	else
	{
		return static_cast<typename copy_cv<T, TOut>::type>(std::forward<T>(obj));
	}
}

#endif //MATHEMATICS_UTILITY_HPP
