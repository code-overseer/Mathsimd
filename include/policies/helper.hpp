
#ifndef MATHEMATICS_HELPER_HPP
#define MATHEMATICS_HELPER_HPP
#include <type_traits>

namespace mathsimd
{
	namespace helper
	{
		template<typename T>
		using is_valid = std::is_base_of <TensorBase<typename T::numeric_type, T::rows, T::cols>, T>;

		template<typename T>
		using sfinae = typename std::enable_if<is_valid<T>::value>::type;

		template<typename T>
		using Bare = typename std::decay<T>::type;
	}
}
#endif //MATHEMATICS_HELPER_HPP
