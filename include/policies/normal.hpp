#ifndef MATHEMATICS_POLICIES_NORMAL_HPP
#define MATHEMATICS_POLICIES_NORMAL_HPP

#include "../bool.hpp"
#include "helper.hpp"
#include <cmath>
#include <numeric>

namespace mathsimd
{
    template<typename T>
    struct CMathPolicy
	{
    	static_assert(std::is_arithmetic<T>::value);
    	struct Vector;
    	struct Matrix;
	};

    template<>
    struct CMathPolicy<float>::Vector
    {
    private:
		static float rsqrt(float number )
		{
			int i{0};
			float x2, y{0};
			static constexpr float threehalfs = 1.5F;

			x2 = number * 0.5F;
			y  = number;
			memcpy(&i, &y, sizeof(y));
			i  = 0x5F1FFFF9 - ( i >> 1 );
			memcpy(&y, &i, sizeof(y));
			y  *= 0.703952253f * ( 2.38924456f - ( x2 * y * y ) );
			y  *= ( threehalfs - ( x2 * y * y ) );

			return y;
		}
    public:
		using numeric_type = float;
        template<typename T>
        static float dot(T const &l, T const &r)
        {
        	auto i0 = static_cast<float const*>(l);
        	auto i1 = static_cast<float const*>(r);
            return std::inner_product(i0, i0 + T::length, i1, 0.0f);
        }

        static float square_root(float val)
        {
            return val * rsqrt(val);
        }
        
    private:
    	template<typename T, size_t L = T::length>
    	struct Cross;
        
        template<typename T>
        struct Cross<T,2>
		{
			static T value(T const &left, T const&right)
			{
				return T(0,0);
			}
		};

		template<typename T>
		struct Cross<T,3>
		{
			static T value(T const &left, T const&right)
			{
				return T(left[1]*right[2] - left[2]*right[1],
						 left[2]*right[0] - left[0]*right[2],
						 left[0]*right[1] - left[1]*right[0]);
			}
		};

		template<typename T>
		struct Cross<T,4>
		{
			static T value(T const &left, T const&right)
			{
				return T(left[1]*right[2] - left[2]*right[1],
						 left[2]*right[0] - left[0]*right[2],
						 left[0]*right[1] - left[1]*right[0],
						 left[3]*right[3] - left[3]*right[3]);
			}
		};
    public:

        template<typename T>
        static T cross(T const &l, T const &r)
		{
			return Cross<T>::value(l,r);
		}

        template<typename T>
        static T minimum(T const &l, T const &r)
        {
            T result;
            for (auto i = 0; i < T::length; ++i)
            {
                result[i] = l[i] < r[i] ? l[i] : r[i];
            }

            return result;
        }

        template<typename T>
        static T maximum(T const &l, T const &r)
        {
            T result;
            for (auto i = 0; i < T::length; ++i)
            {
                result[i] = l[i] > r[i] ? l[i] : r[i];
            }

            return result;
        }

        template<typename T>
        static T sign(T const &val)
        {
            T result;
            for (auto i = 0; i < T::length; ++i)
            {
                result[i] = val[i] > 0 ? 1 : val[i] < 0 ? -1 : 0;
            }

            return result;
        }

		template<typename T>
		static T absolute(T const& val)
		{
			T result;
			for (auto i = 0; i < T::length; ++i)
			{
				result[i] = std::fabs(val[i]);
			}

			return result;
		}

		template<typename T>
		static T normalize(T const &val)
		{
			return val * rsqrt(dot(val, val));
		}

		template<typename T>
		static T reciprocal(T const& val)
		{
			T result;
			for (auto i = 0; i < T::length; ++i)
			{
				result[i] = rsqrt(val[i]);
			}
			return multiply(result, result);
		}

        #define ARITHMETIC(NAME, SYM) \
        template<typename T> \
        static T NAME(T const &l, T const &r) \
        { \
            T result; \
            for (auto i = 0; i < T::length; ++i) \
            { \
                result[i] = l[i] SYM r[i]; \
            } \
            return result; \
        } \
        template<typename T> \
        static T NAME(float l, T const &r) \
        { \
            T result; \
            for (auto i = 0; i < T::length; ++i) \
            { \
                result[i] = l SYM r[i]; \
            } \
            return result; \
        } \
        template<typename T> \
        static T NAME(T const &l, float r) \
        { \
            T result; \
            for (auto i = 0; i < T::length; ++i) \
            { \
                result[i] = l[i] SYM r; \
            } \
            return result; \
        } 

        ARITHMETIC(add,+)
        ARITHMETIC(subtract,-)
        ARITHMETIC(multiply,*)
        ARITHMETIC(divide,/)

        #undef ARITHMETIC

        #define CMP(NAME, SYM) \
        template<typename T> \
        static Bool<T::length> NAME(T const &l, T const &r) \
        { \
            Bool<T::length> result; \
            for (auto i = 0; i < T::length; ++i) \
            { \
                if (l[i] SYM r[i]) result.set(i); \
            } \
            return result; \
        } \
        template<typename T> \
        static Bool<T::length> NAME(float l, T const &r) \
        { \
            Bool<T::length> result; \
            for (auto i = 0; i < T::length; ++i) \
            { \
                if (l SYM r[i]) result.set(i); \
            } \
            return result; \
        } \
        template<typename T> \
        static Bool<T::length> NAME(T const &l, float r) \
        { \
            Bool<T::length> result; \
            for (auto i = 0; i < T::length; ++i) \
            { \
                if (l[i] SYM r) result.set(i); \
            } \
            return result; \
        }

        CMP(less , <)
        CMP(less_equal, <=)
        CMP(greater, >)
        CMP(greater_equal, >=)
        CMP(equal, ==)
        CMP(not_equal, !=)

        #undef CMP
    };

}

#endif // MATHEMATICS_OPERATIONS_NORMAL_HPP
