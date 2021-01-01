#ifndef MATHEMATICS_REGISTER_POLICY_HPP
#define MATHEMATICS_REGISTER_POLICY_HPP
#include "../utility.hpp"

namespace mathsimd
{
	template<typename T, typename Reg>
	struct RegisterPolicy
	{
		using Register = Reg;
		static size_t constexpr alignment()
		{
			return alignof(T);
		}
		static size_t constexpr active_size()
		{
			return T::active_bytes();
		}
		static size_t constexpr register_size()
		{
			return sizeof(Register);
		}
		static size_t constexpr active_aligned()
		{
			// division to deal with larger alignments
			return T::active_aligned_bytes() / utility::not_zero(alignment()/alignof(Register));
		}
		static bool constexpr is_aligned() { return !(alignment() % alignof(Register)); }
		template<typename U>
		static size_t constexpr alignment() { return utility::count<U>(alignment()); };
		template<typename U>
		static size_t constexpr active_size() { return utility::count<U>(active_size()); };
		template<typename U>
		static size_t constexpr register_size() { return utility::count<U>(register_size()); }
		template<typename U>
		static size_t constexpr active_aligned() { return utility::count<U>(active_aligned()); }
	};
}

#endif //MATHEMATICS_REGISTER_POLICIES_HPP
