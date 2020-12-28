#ifndef MATHEMATICS_REGISTER_POLICY_HPP
#define MATHEMATICS_REGISTER_POLICY_HPP

namespace mathsimd
{
	template<typename T, typename Reg>
	struct RegisterPolicy
	{
	protected:
		static size_t constexpr ceil_div(size_t left, size_t right) { return 1 + ((left - 1) / right); }
		static bool constexpr is_pow2(size_t value) { return !(value & (value - 1)); }
		static float const* cast(T const& v) { return v; }
	public:
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
			return T::active_aligned_bytes() / ceil_div(alignment(), alignof(Register));
		}
		static bool constexpr is_aligned() { return alignment() == alignof(Register); }
		template<typename U>
		static size_t constexpr alignment() { return alignment() / sizeof(U); };
		template<typename U>
		static size_t constexpr active_size() { return active_size() / sizeof(U); };
		template<typename U>
		static size_t constexpr register_size() { return register_size() / sizeof(U); }
		template<typename U>
		static size_t constexpr active_aligned() { return active_aligned() / sizeof(U); }
	};
}

#endif //MATHEMATICS_REGISTER_POLICIES_HPP
