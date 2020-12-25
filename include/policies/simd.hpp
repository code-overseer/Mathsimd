#ifndef MATHEMATICS_POLICIES_SIMD_HPP
#define MATHEMATICS_POLICIES_SIMD_HPP

#include "../bool.hpp"
#include "../tensor_base.hpp"
#include "helper.hpp"
#include "../array.hpp"
#include <immintrin.h>
#include <utility>

inline __m128 _mm_abs_ps(__m128 const& fp_val)
{
	__m128 mask = _mm_castsi128_ps(_mm_set1_epi32((((1 << 30) - 1) << 1) + 1));
	return _mm_and_ps(fp_val, mask);
}

inline __m256 _mm256_abs_ps(__m256 const& fp_val)
{
	__m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32((((1 << 30) - 1) << 1) + 1));
	return _mm256_and_ps(fp_val, mask);
}

namespace mathsimd
{
	template<typename T>
    struct SimdVectorPolicy
    {
        struct M128;
    };
	
	
	template<>
    struct SimdVectorPolicy<float>::M128
    {
    private:
        static inline __m128 _dot(__m128 left, __m128 right)
        {
            auto tmp = _mm_mul_ps(left, right);
            tmp = _mm_add_ps(tmp, _mm_permute_ps(tmp, _MM_SHUFFLE(1,0,3,2)));
            return _mm_add_ss(tmp, _mm_permute_ps(tmp, _MM_SHUFFLE(2,3,0,1)));
        }

        static inline __m128 _sqrt(__m128 val)
        {
            return _mm_mul_ss(val, _mm_rsqrt_ss(val));
        }
    public:
    	using numeric_type = float;
        template<typename T>
        static float dot(T const &l, T const &r)
        {
            float f;
            _mm_store_ss(&f, _dot(l, r));
            return f;
        }

        template<typename T>
        static T cross(T const &l, T const &r)
        {
            constexpr int mask0 = _MM_SHUFFLE(3,0,2,1);
            constexpr int mask1 = _MM_SHUFFLE(3,1,0,2);
            auto tmp0 = _mm_permute_ps(l,mask0);
            auto tmp1 = _mm_permute_ps(r,mask1);
            auto tmp2 = _mm_permute_ps(l,mask1);
            auto tmp3 = _mm_permute_ps(r,mask0);
            return _mm_fmsub_ps(tmp0, tmp1, _mm_mul_ps(tmp2, tmp3));
        }

		template<typename T>
		static T normalize(T const &val)
		{
			auto tmp = _dot(val, val);
			tmp = _mm_permute_ps(tmp, _MM_SHUFFLE(0,0,0,0));
			return _mm_mul_ps(val, _mm_rsqrt_ps(tmp));
		}

		static float square_root(float val)
		{
			auto reg = _mm_set_ss(val);
			_mm_store_ss(&val, _sqrt(reg));
			return val;
		}

        template<typename T>
        static T minimum(T const &l, T const &r)
        {
            auto c = _mm_cmplt_ps(l, r);
            return _mm_or_ps(_mm_and_ps(c,l), _mm_andnot_ps(c,r));
        }

        template<typename T>
        static T maximum(T const &l, T const &r)
        {
            auto c = _mm_cmpgt_ps(l, r);
            return _mm_or_ps(_mm_and_ps(c,l), _mm_andnot_ps(c,r));
        }

        template<typename T>
        static T sign(T const &val)
        {
            static constexpr float plus = 1.0f;
            static constexpr float minus = -1.0f;
            __m128 zero = _mm_setzero_ps();

            __m128 positive = _mm_and_ps(_mm_cmpgt_ps(val, zero), _mm_broadcast_ss(&plus));
            __m128 negative = _mm_and_ps(_mm_cmplt_ps(val, zero), _mm_broadcast_ss(&minus));

            return _mm_or_ps(positive, negative);
        }

        template<typename T>
        static T reciprocal(T const &val)
        {
            return _mm_rcp_ps(val);
        }

		template<typename T>
		static T absolute(T const& val)
		{
			return _mm_abs_ps(val);
		}

        #define ARITHMETIC(FUNC, OP) \
        template<typename T> \
        static T FUNC(T const &l, T const &r) \
        { \
            return _mm_ ## OP ## _ps(l,r); \
        } \
        template<typename T> \
        static T FUNC(float const &l, T const &r) \
        { \
            return _mm_ ## OP ## _ps(_mm_broadcast_ss(&l),r); \
        } \
        template<typename T> \
        static T FUNC(T const & l, float const &r) \
        { \
            return _mm_ ## OP ## _ps(l, _mm_broadcast_ss(&r)); \
        }

        ARITHMETIC(add,add)
        ARITHMETIC(subtract,sub)
        ARITHMETIC(multiply,mul)
        ARITHMETIC(divide,div)

        #undef ARITHMETIC

        #define CMP(FUNC, OP) \
        template<typename T> \
        static Bool<T::length> FUNC(T const &l, T const &r) \
        { \
            return _mm_movemask_ps(_mm_ ## OP ## _ps(l,r)); \
        } \
        template<typename T> \
        static Bool<T::length> FUNC(typename T::type l, T const &r) \
        { \
            return _mm_movemask_ps(_mm_ ## OP ## _ps(_mm_broadcast_ss(&l),r)); \
        } \
        template<typename T> \
        static Bool<T::length> FUNC(T const & l, typename T::type r) \
        { \
            return _mm_movemask_ps(_mm_ ## OP ## _ps(l, _mm_broadcast_ss(&r))); \
        }

        CMP(less,cmplt)
        CMP(less_equal,cmple)
        CMP(greater,cmpgt)
        CMP(greater_equal,cmpge)
        CMP(equal,cmpeq)
        CMP(not_equal,cmpneq)

        #undef CMP
    };

	template<typename T>
	struct Simd4x4Policy;

	template<>
	struct Simd4x4Policy<float>
	{
	private:
		template<typename T>
		struct loader
		{
			static_assert(!(alignof(T) % 32));
			template<size_t N>
			static inline __m256 load(T const& matrix)
			{
				return _mm256_load_ps(static_cast<float const*>(matrix) + N * 8);
			}
		};
	public:

		template<typename T>
		static T minimum(T const &left, T const &right)
		{
			auto l = loader<T>::load<0>(left);
			auto r = loader<T>::load<0>(right);
			auto upper = _mm256_or_ps(_mm256_and_ps(_mm256_cmplt_ps(l, r), l));
			l = loader<T>::load<1>(left);
			r = loader<T>::load<1>(right);
			return {upper, _mm256_or_ps(_mm256_and_ps(_mm256_cmplt_ps(l, r), l))};
		}

		template<typename T>
		static T maximum(T const &left, T const &right)
		{
			auto l = loader<T>::load<0>(left);
			auto r = loader<T>::load<0>(right);
			auto upper = _mm256_or_ps(_mm256_and_ps(_mm256_cmpgt_ps(l, r), l));
			l = loader<T>::load<1>(left);
			r = loader<T>::load<1>(right);
			return {upper, _mm256_or_ps(_mm256_and_ps(_mm256_cmpgt_ps(l, r), l))};
		}

		template<typename T>
		static T sign(T const &val)
		{
			static constexpr float _plus = 1.0f;
			static constexpr float _minus = -1.0f;
			__m256 plus = _mm256_broadcast_ss(&_plus);
			__m256 minus = _mm256_broadcast_ss(&_minus);
			__m256 zero = _mm256_setzero_ps();

			__m256 positive = _mm256_and_ps(_mm256_cmpgt_ps(loader<T>::load<0>(val), zero), plus);
			__m256 negative = _mm256_and_ps(_mm256_cmplt_ps(loader<T>::load<0>(val), zero), minus);
			__m256 upper = _mm256_or_ps(positive, negative);
			positive = _mm256_and_ps(_mm256_cmpgt_ps(loader<T>::load<1>(val), zero), plus);
			negative = _mm256_and_ps(_mm256_cmplt_ps(loader<T>::load<1>(val), zero), minus);

			return {upper, _mm256_or_ps(positive, negative)};
		}

		template<typename T>
		static T reciprocal(T const &val)
		{
			return {_mm256_rcp_ps(loader<T>::load<0>(val)),	_mm256_rcp_ps(loader<T>::load<1>(val))};
		}

		template<typename T>
		static T absolute(T const& val)
		{
			return {_mm256_abs_ps(loader<T>::load<0>(val)),	_mm256_abs_ps(loader<T>::load<1>(val))};
		}

		#define ARITHMETIC(FUNC, OP) \
        template<typename T> \
		static T FUNC(T const& left, T const & right) \
		{ \
			return {_mm256_ ## OP ## _ps(loader<T>::load<0>(left), loader<T>::load<0>(right)), _mm256_ ## OP ## _ps(loader<T>::load<1>(left), loader<T>::load<1>(right))}; \
		} \
		template<typename T> \
        static T FUNC(typename T::type left, T const &right) \
 		{ \
 			auto tmp = _mm256_set1_ps(left); \
 			return { _mm256_ ## OP ## _ps(tmp, loader<T>::load<0>(right)), _mm256_ ## OP ## _ps(tmp, loader<T>::load<1>(right)) }; \
		} \
		template<typename T> \
        static T FUNC(T const &left, typename T::type right) \
 		{ \
 			auto tmp = _mm256_set1_ps(right); \
 			return { _mm256_ ## OP ## _ps(loader<T>::load<0>(left), tmp), _mm256_ ## OP ## _ps(loader<T>::load<1>(left), tmp) }; \
		}

		ARITHMETIC(add,add)
		ARITHMETIC(subtract,sub)
		ARITHMETIC(multiply,mul)
		ARITHMETIC(divide,div)

		#undef ARITHMETIC

		#define CMP(FUNC, OP) \
        template<typename T> \
        static Bool<16> FUNC(T const &left, T const &right) \
        { \
        	auto result0 = _mm256_movemask_ps(_mm256_ ## OP ## _ps(loader<T>::load<0>(left), loader<T>::load<0>(right))); \
        	auto result1 = _mm256_movemask_ps(_mm256_ ## OP ## _ps(loader<T>::load<1>(left), loader<T>::load<1>(right))); \
            return (result1 << 8) | result0; \
        } \
        template<typename T> \
        static Bool<16> FUNC(typename T::type left, T const &right) \
        { \
        	auto tmp = _mm256_set1_ps(left); \
            auto result0 = _mm256_movemask_ps(_mm256_ ## OP ## _ps(tmp, loader<T>::load<0>(right))); \
        	auto result1 = _mm256_movemask_ps(_mm256_ ## OP ## _ps(tmp, loader<T>::load<1>(right))); \
            return (result1 << 8) | result0; \
        } \
        template<typename T> \
        static Bool<16> FUNC(T const & left, typename T::type right) \
        { \
        	auto tmp = _mm256_set1_ps(right); \
            auto result0 = _mm256_movemask_ps(_mm256_ ## OP ## _ps(loader<T>::load<0>(left), tmp)); \
        	auto result1 = _mm256_movemask_ps(_mm256_ ## OP ## _ps(loader<T>::load<1>(left), tmp)); \
            return (result1 << 8) | result0; \
        }

		CMP(less,cmplt)
		CMP(less_equal,cmple)
		CMP(grater,cmpgt)
		CMP(greater_equal,cmpge)
		CMP(equal,cmpeq)
		CMP(not_equal,cmpneq)

		#undef CMP
	};
}

#endif // MATHEMATICS_OPERATIONS_SIMD_HPP
