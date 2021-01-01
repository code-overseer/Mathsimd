#ifndef MATHEMATICS_CHUNKER_HPP
#define MATHEMATICS_CHUNKER_HPP

namespace mathsimd
{
	template<size_t StructSize, size_t Length, size_t Target, bool Early = Length * StructSize <= Target, bool Recurse = Length * StructSize % Target>
	struct Chunker;

	template<size_t StructSize, size_t Length, size_t Target, bool Recurse>
	struct Chunker<StructSize, Length, Target, true, Recurse>
	{
		static size_t constexpr length()
		{
			return Length;
		}
		static size_t constexpr count()
		{
			return 1;
		}

		static size_t index(size_t const idx)
		{
			return idx;
		}
	};

	template<size_t StructSize, size_t Length, size_t Target>
	struct Chunker<StructSize, Length, Target, false, true> : Chunker<StructSize, Length, Target - StructSize, false>{};

	template<size_t StructSize, size_t Length, size_t Target>
	struct Chunker<StructSize, Length, Target, false, false>
	{
		static size_t constexpr length()
		{
			return utility::not_zero(Target / StructSize);
		}
		static size_t constexpr count()
		{
			return utility::not_zero(Length / length());
		}
		static size_t index(size_t const idx)
		{
			if constexpr (length() == 1)
			{
				return idx;
			}

			return idx + idx / length();
		}
	};
}
#endif //MATHEMATICS_CHUNKER_HPP
