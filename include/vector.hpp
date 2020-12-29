#ifndef MATHEMATICS_VECTOR_HPP
#define MATHEMATICS_VECTOR_HPP
#include "bool.hpp"
#include "matrix_base.hpp"

namespace mathsimd
{
	/// Do not use! This is just a code generator for float2, float3, float4
	/// It assumes the size of the vector is <= alignment (NOT the case for Length > 4)
	template<size_t Length, size_t Alignment>
	struct Float
	{
		using numeric_type = float;
		static_assert(Length <= 4 && Length * sizeof(numeric_type) <= Alignment);
	protected:
		alignas(Alignment) float values_[Length]{0.0f};
	public:
		Float() = default;
		Float(float const&x, float const& y,float const& z,float const& w) : values_{x,y,z,w} {}
		static size_t constexpr length() { return Length; }
		static size_t constexpr alignment() { return Alignment; }
		static size_t constexpr active_aligned_bytes() { return length() * sizeof(numeric_type); }
		static size_t constexpr active_bytes() { return length() * sizeof(numeric_type); }
		inline operator float*() { return values_; }
		inline operator float const*() const { return values_; }
		float operator[](size_t const idx) const { return values_[idx]; }
		float& operator[](size_t const idx) { return values_[idx]; }
	};

	using float2 = Float<2,8>;
	using float3 = Float<3,16>;
	using float4 = Float<4,16>;

	template<typename T, template<size_t,size_t> typename TBitField, template<typename> typename Policy>
	struct Vector : T, MatrixBase<T::length(), 1>, private Policy<T>
	{
		using T::length;
		using T::alignment;
	private:
		using OpPolicy = Policy<T>;
		using Boolean = TBitField<length(), alignment()>;
	public:
		using T::T;
		using T::operator[];
		using T::operator typename T::numeric_type*;
		using T::operator typename T::numeric_type const*;
		Vector(Vector const&other) { OpPolicy::Copy(*this, other); }
		Vector& operator=(Vector const& other)
		{
			if (this != &other)
			{
				OpPolicy::Copy(*this, other);
			}

			return *this;
		}
		#define OPERATOR_DEFINE(OP,FUNC,LEFT,RIGHT) \
		friend decltype(auto) operator OP(LEFT const& left, RIGHT const& right) \
		{ \
			return OpPolicy::FUNC(left, right); \
		}

		#define OPERATOR(SYM,FUNC) \
		OPERATOR_DEFINE(SYM,FUNC,Vector,Vector) \
		OPERATOR_DEFINE(SYM,FUNC,Vector,typename T::numeric_type) \
		OPERATOR_DEFINE(SYM,FUNC,typename T::numeric_type,Vector)

		OPERATOR(+,add)
		OPERATOR(-,subtract)
		OPERATOR(*,multiply)
		OPERATOR(/,divide)
		OPERATOR(<,template less<Boolean>)
		OPERATOR(>,template greater<Boolean>)
		OPERATOR(<=,template less_equals<Boolean>)
		OPERATOR(>=,template greater_equals<Boolean>)
		OPERATOR(==,template equals<Boolean>)
		OPERATOR(!=,template not_equals<Boolean>)

		#undef OPERATOR
		#undef OPERATOR_DEFINE

		#define UNARY_OPS(FUNC) \
		friend decltype(auto) FUNC(Vector const& vector) { return OpPolicy::FUNC(vector); };

		UNARY_OPS(reciprocal)
		UNARY_OPS(sum)
		UNARY_OPS(difference)
		UNARY_OPS(minimum)
		UNARY_OPS(maximum)
		UNARY_OPS(absolute)
		UNARY_OPS(sign)
		UNARY_OPS(rsqrt)
		UNARY_OPS(sqrt)
		#undef UNARY_OPS

		#define BINARY_OPS(FUNC) \
		friend decltype(auto) FUNC(Vector const& left, Vector const& right) { return OpPolicy::FUNC(left, right); };

		BINARY_OPS(sum_product)
		BINARY_OPS(diff_product)
		#undef BINARY_OPS

		friend float magnitude(Vector const& vector)
		{
			auto tmp = sum_product(vector, vector);
			return tmp * OpPolicy::rsqrt(tmp);
		}

		friend float sqr_magnitude(Vector const& vector)
		{
			return sum_product(vector, vector);
		}

		friend float dot(Vector const& left, Vector const& right)
		{
			auto tmp = sum_product(left, right);
			return tmp * OpPolicy::rsqrt(tmp);
		}
	};

	template<template<typename> typename Policy>
	using Float2 = Vector<float2, Bool, Policy>;
	template<template<typename> typename Policy>
	using Float3 = Vector<float3, Bool, Policy>;
	template<template<typename> typename Policy>
	using Float4 = Vector<float4, Bool, Policy>;
}

#endif //MATHEMATICS_VECTOR_HPP
