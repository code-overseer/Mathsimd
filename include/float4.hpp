#ifndef MATHEMATICS_SIMD_FLOAT4_HPP
#define MATHEMATICS_SIMD_FLOAT4_HPP

#include <immintrin.h>
#include "constants.hpp"
#include "vector_operations.hpp"
#include "float2.hpp"
#include "float3.hpp"
#include "bool.hpp"

namespace mathsimd 
{
    struct float4 : TensorBase<float, 4, 1>, private Operations<float4, SimdVectorPolicy<float>::M128>
    {
    protected:
        alignas(16) float _val[4]{0.f, 0.f, 0.f, 0.f};
    public:

        float4() = default;
		float4(float (&other)[4]) { _mm_store_ps(_val, _mm_load_ps(other)); }
		float4(float const &x, float const &y, float const &z, float const &w) : _val{x, y, z, w} {}
        float4(float const &x, float3 const &yzw) : _val{x, yzw.x(), yzw.y(), yzw.z()} {}
        float4(float3 const &xyz, float const &w) : _val{xyz.x(), xyz.y(), xyz.z(), w} {}
        float4(float const &x, float const &y, float2 const &zw) : _val{x, y, zw.x(), zw.y()} {}
        float4(float2 const &xy, float const &z, float const &w) : _val{xy.x(), xy.y(), z, w} {}
        float4(float const &x, float2 const &yz, float const &w) : _val{x, yz.x(), yz.y(), w} {}
        float4(float4 const &other) { _mm_store_ps(_val, _mm_load_ps(other._val)); }
        float4(float const* other) { _mm_storeu_ps(_val, _mm_loadu_ps(other));}
        float4(__m128 const &other) { _mm_store_ps(_val, other); }
        float4(float const &x) { _mm_store_ps(_val, _mm_broadcast_ss(&x)); }
        inline operator float*() { return _val; }
        inline operator float const*() const { return _val; }
        inline operator __m128() const { return _mm_load_ps(_val); }
        inline float4 &operator=(float4 const &other) = default;
        inline float4 &operator=(__m128 const &other) { _mm_store_ps(_val, other); return *this; }
        inline float operator[](size_t const idx) const { return _val[idx]; }
        inline float& operator[](size_t const idx) { return _val[idx]; }
        
        float &x() { return _val[0]; }
        float &y() { return _val[1]; }
        float &z() { return _val[2]; }
        float &w() { return _val[3]; }
        [[nodiscard]] float x() const { return _val[0]; }
        [[nodiscard]] float y() const { return _val[1]; }
        [[nodiscard]] float z() const { return _val[2]; }
        [[nodiscard]] float w() const { return _val[3]; }

        #define FUNC(NAME,X,Y,Z,W) \
        static inline float4 NAME () { return {X,Y,Z,W}; }
        FUNC(up, 0,1,0,0)
        FUNC(down, 0,-1,0,0)
        FUNC(right, 1,0,0,0)
        FUNC(left, -1,0,0,0)
        FUNC(forward, 0,0,1,0)
        FUNC(back, 0,0,-1,0)
        FUNC(in, 0,0,0,1)
        FUNC(out, 0,0,0,-1)
        FUNC(one, 1,1,1,1)
        FUNC(zero, 0,0,0,0)
        FUNC(origin, 0,0,0,1)
        #undef FUNC
    };

	template<typename OperationPolicy>
	struct Float4 : float4, private Operations<Float4<OperationPolicy>, OperationPolicy>
	{
		using base_type = float4;
		using float4::float4;
		REBINDER(Float4, OperationPolicy)
	};
}
#endif //MATHEMATICS_SIMD_FLOAT4_HPP