#ifndef MATHEMATICS_VECTOR_HPP
#define MATHEMATICS_VECTOR_HPP

#include "bool.hpp"
#include "utility.hpp"
#include "matrix_base.hpp"

namespace mathsimd
{
	using _float2 = _Matrix<2,1,8,8>;
	using _float3 = _Matrix<3,1,16,16>;
	using _float4 = _Matrix<4,1,16,16>;

	template<typename T, template<size_t,size_t> typename TBitField, template<typename> typename Policy>
	struct Vector : T, private Policy<T>
	{
		using Base = T;
		static_assert(Base::columns() == 1);
		using numeric_type = typename Base::numeric_type;
		using T::length;
		using T::alignment;
		using OpPolicy = Policy<T>;
		using Boolean = TBitField<length(), utility::count<numeric_type>(alignment())>;
		using T::T;
		using T::operator[];
		using T::operator numeric_type*;
		using T::operator numeric_type const*;
		Vector() = default;
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
		OPERATOR_DEFINE(SYM,FUNC,Vector,numeric_type) \
		OPERATOR_DEFINE(SYM,FUNC,numeric_type,Vector)

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
	using float2 = Vector<_float2, Bool, Policy>;
	template<template<typename> typename Policy>
	using float3 = Vector<_float3, Bool, Policy>;
	template<template<typename> typename Policy>
	using float4 = Vector<_float4, Bool, Policy>;


}

#endif //MATHEMATICS_VECTOR_HPP
