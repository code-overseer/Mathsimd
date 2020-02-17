#include "../../include/seq/float3.hpp"

float mathseq::float3::dot(const mathseq::float3 &a, const mathseq::float3 &b) {
    return std::inner_product(a._val.begin(), a._val.end(), b._val.begin(), 0.f);
}

mathseq::float3 mathseq::float3::cross(const mathseq::float3 &a, const mathseq::float3 &b) {
    return float3{
            a.y * b.z - b.y * a.z,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    };
}
