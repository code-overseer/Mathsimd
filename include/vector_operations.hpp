#ifndef MATHEMATICS_VECTOR_OPERATIONS_HPP
#define MATHEMATICS_VECTOR_OPERATIONS_HPP

#include "tensor_base.hpp"
#include "bool.hpp"
#include <type_traits>
#include <ostream>

namespace mathsimd 
{
    template<typename TVector, typename OperationPolicy>
    struct Operations
    {
    private:
    	using numeric_type = typename OperationPolicy::numeric_type;
		using Vector = TVector;
	public:
		friend float dot(Vector const &left, Vector const &right)
		{
			return OperationPolicy::dot(left, right);
		}

		friend Vector normalize(Vector const& vector)
		{
			return OperationPolicy::normalize(vector);
		}

		friend float sqr_magnitude(Vector const& vector)
		{
			return OperationPolicy::dot(vector, vector);
		}

		friend float magnitude(Vector const& vector)
		{
			return OperationPolicy::square_root(sqr_magnitude(vector));
		}

		friend Vector cross(Vector const& left, Vector const & right)
		{
			return OperationPolicy::cross(left, right);
		}

		friend Vector minimum(Vector const& left, Vector const & right)
		{
			return OperationPolicy::minimum(left, right);
		}

		friend Vector maximum(Vector const& left, Vector const & right)
		{
			return OperationPolicy::maximum(left, right);
		}

		friend Vector absolute(Vector const& vector)
		{
			return OperationPolicy::absolute(vector);
		}

		friend Vector sign(Vector const& vector)
		{
			return OperationPolicy::sign(vector);
		}

		friend Vector reciprocal(Vector const& vector)
		{
			return OperationPolicy::reciprocal(vector);
		}

		friend std::ostream& operator<<(std::ostream& stream, Vector const& vector)
		{
			stream << "(" << vector[0];
			for (auto i = 1; i < Vector::length; ++i)
			{
				stream << ", " << vector[i];
			}
			return stream << ")";
		}

		#define ARITHMETIC(OP,SYM) \
		friend Vector operator SYM (Vector const& left, Vector const & right) \
		{ \
			return OperationPolicy::OP(left, right); \
		} \
		friend Vector operator SYM (numeric_type const &left, Vector const &right) \
		{ \
			return OperationPolicy::OP(left, right); \
		} \
		friend Vector operator SYM (Vector const &left, numeric_type const &right) \
		{ \
			return OperationPolicy::OP(left, right); \
		}

		ARITHMETIC(add, +)

		ARITHMETIC(subtract, -)

		ARITHMETIC(multiply, *)

		ARITHMETIC(divide, /)

		#undef ARITHMETIC

		#define CMP(OP,SYM) \
		friend auto operator SYM (Vector const& left, Vector const& right) \
		{ \
			return OperationPolicy::OP(left, right); \
		} \
		friend auto operator SYM (numeric_type const &left, Vector const &right) \
		{ \
			return OperationPolicy::OP(left, right); \
		} \
		friend auto operator SYM (Vector const &left, numeric_type const &right) \
		{ \
			return OperationPolicy::OP(left, right); \
		}

		CMP(less, <)

		CMP(less_equal, <=)

		CMP(greater, >)

		CMP(greater_equal, >=)

		CMP(equal, ==)

		CMP(not_equal, !=)
		#undef CMP
    };

    #define REBINDER(NAME,POLICY) \
    template<typename UOperations> \
    struct rebind \
    { \
        using type = NAME<UOperations>; \
    }; \
    template<typename U> \
    inline operator U() const \
    {  \
        static_assert(std::is_same<NAME<POLICY>, typename U::template rebind<POLICY>::type>::value); \
        return {_val}; \
    } \
    template<typename UOperations> \
    inline typename rebind<UOperations>::type cast() const { return *this; };
}

#endif //MATHEMATICS_VECTOR_OPERATIONS_HPP
