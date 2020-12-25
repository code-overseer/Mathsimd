#ifndef MATHEMATICS_ARRAY_HPP
#define MATHEMATICS_ARRAY_HPP

namespace mathsimd
{
	template<typename T, size_t N>
	struct Array
	{
		T value[N];

		Array() = default;

		template<typename... U>
		Array(U&&... args) : value{args...}
		{
		}

		Array(Array const& other) = default;
		Array(Array && other) = default;

		inline operator T*() { return value; }
		inline operator T const *() const { return value; }
		inline T& operator[](size_t const idx) { return value[idx]; }
		inline T const & operator[](size_t const idx) const { return value[idx]; }
	};
}
#endif //MATHEMATICS_ARRAY_HPP
