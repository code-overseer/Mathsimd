#ifndef MATHEMATICS_SIMD_FLOAT3_HPP
#define MATHEMATICS_SIMD_FLOAT3_HPP

#include <immintrin.h>
#include "vector_operations.hpp"
#include "float2.hpp"
#include "constants.hpp"
#include "bool.hpp"

namespace mathsimd
{
    struct float3 : TensorBase<float, 3, 1>, private Operations<float3, SimdVectorPolicy<float>::M128>
	{
	protected:
        alignas(16) float _val[3]{0, 0, 0};
    public:
        float3() = default;
		float3(float (&other)[3]) { _mm_store_ps(_val, _mm_load_ps(other)); }
        float3(float const &x, float const &y, float const &z) : _val{x, y, z} {}
        float3(float2 const &xy, float const &z) : _val{xy.x(), xy.y(), z} {}
        float3(float const &x, float2 const &yz) : _val{x, yz.x(), yz.y()} {}
        float3(float const* other) { _mm_storeu_ps(_val, _mm_loadu_ps(other)); }
        float3(float3 const &other) { _mm_store_ps(_val, _mm_load_ps(other._val)); }
        float3(__m128 const &other) { _mm_store_ps(_val, other); }
        float3(float const &x) { _mm_store_ps(_val, _mm_broadcast_ss(&x)); }
        inline operator float*() { return _val; }
        inline operator float const*() const { return _val; }
        inline operator __m128() const { return _mm_load_ps(_val); }
        inline float3 &operator=(float3 const &other) = default;
        inline float3 &operator=(__m128 const &other) { _mm_store_ps(_val, other); return *this; }
        inline float operator[](size_t const idx) const { return _val[idx]; }
        inline float& operator[](size_t const idx) { return _val[idx]; }

        float &x() { return _val[0]; }
        float &y() { return _val[1]; }
        float &z() { return _val[2]; }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }
        [[nodiscard]] float z() const { return _val[2]; }

        #define FUNC(NAME,X,Y,Z) \
        static inline float3 NAME () { return {X,Y,Z}; }
        FUNC(up, 0,1,0)
        FUNC(down, 0,-1,0)
        FUNC(right, 1,0,0)
        FUNC(left, -1,0,0)
        FUNC(forward, 0,0,1)
        FUNC(back, 0,0,-1)
        FUNC(one, 1,1,1)
        FUNC(zero, 0,0,0)
        #undef FUNC
    };

	template<typename OperationPolicy>
	struct Float3 : float3, private Operations<Float3<OperationPolicy>, OperationPolicy>
	{
		using base_type = float3;
		using float3::float3;
		REBINDER(Float3, OperationPolicy)
	};
}
#endif //MATHEMATICS_SIMD_FLOAT3_HPP