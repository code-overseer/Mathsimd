#ifndef MATHEMATICS_TENSOR_BASE_HPP
#define MATHEMATICS_TENSOR_BASE_HPP

#include <type_traits>
#include "utility.hpp"
#include "chunker.hpp"

namespace mathsimd
{
	/// Non POD, not safe to reinterpret_cast
	template<size_t Rows, size_t Columns, size_t ColumnAlignment, size_t Alignment>
	struct _Matrix
	{
		using numeric_type = float;
		static_assert(Rows && Columns && ColumnAlignment && Alignment &&
						utility::is_pow2(ColumnAlignment) && utility::is_pow2(Alignment) &&
					  	Alignment >= ColumnAlignment);

		static size_t constexpr rows() { return Rows; }
		static size_t constexpr columns() { return Columns; }
		static size_t constexpr length() { return Rows * Columns; }
		static size_t constexpr column_alignment() { return ColumnAlignment; }
		static size_t constexpr alignment() { return Alignment; }
	protected:
		using ColChunk = Chunker<sizeof(numeric_type), rows(), column_alignment()>;
		using Chunk = Chunker<ColChunk::count() * column_alignment(), columns(), alignment()>;
		alignas(Alignment) numeric_type data_[utility::count<numeric_type>(alignment()) * Chunk::count()]{0.0f};
	public:
		static size_t constexpr active_aligned_bytes() { return utility::size<numeric_type>(ColChunk::length()) * Chunk::length(); }
		static size_t constexpr active_bytes() { return utility::size<numeric_type>(length()); }
		inline operator numeric_type*() { return data_; }
		inline operator numeric_type const*() const { return data_; }
		inline numeric_type* column(size_t const idx) { return data_ + utility::size<float>(Chunk::index(idx)); }
		[[nodiscard]] inline numeric_type const* column(size_t const idx) const { return data_ + utility::size<float>(Chunk::index(idx)); }
		inline numeric_type& operator[](size_t const idx) { return column(idx / Columns) + ColChunk::index(idx % Rows); }
		inline numeric_type operator[](size_t const idx) const { return column(idx / Columns) + ColChunk::index(idx % Rows); }
	};

}

#endif //MATHEMATICS_TENSOR_BASE_HPP
