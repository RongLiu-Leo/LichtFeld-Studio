/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Compiled with --fmad=false: the CPU decoders round every multiply and add
// separately, and contraction here would break bit-parity for the pure
// arithmetic encodings (r8, s8, meta dequant frames).

#include "lod_page_dequant_cuda.hpp"

#include "core/cuda/sh_layout.cuh"
#include "io/formats/rad_dequant_math.hpp"

namespace lfs::vis {
    namespace {

        using lfs::io::RadPackedEncoding;
        using lfs::io::RadPackedKind;
        using lfs::io::RadPackedProperty;
        using lfs::io::RadPagePackedDesc;
        namespace radmath = lfs::io::radmath;

        // Dimension-major plane read for splat i, dimension d, with the same
        // per-element math as rad.cpp's PropertyDecoder dispatch.
        __device__ float readPlane(const std::uint8_t* const plane,
                                   const RadPackedProperty& prop,
                                   const std::uint32_t dims,
                                   const std::uint32_t count,
                                   const std::uint32_t d,
                                   const std::uint32_t i) {
            const std::uint32_t e = d * count + i;
            switch (static_cast<RadPackedEncoding>(prop.encoding)) {
            case RadPackedEncoding::F32: {
                std::uint32_t bits;
                memcpy(&bits, plane + static_cast<std::size_t>(e) * 4u, 4u);
                return radmath::bitsToFloat(bits);
            }
            case RadPackedEncoding::F32LeBytes: {
                const std::uint32_t stride = count * dims;
                const std::uint32_t bits =
                    static_cast<std::uint32_t>(plane[e]) |
                    (static_cast<std::uint32_t>(plane[stride + e]) << 8) |
                    (static_cast<std::uint32_t>(plane[2u * stride + e]) << 16) |
                    (static_cast<std::uint32_t>(plane[3u * stride + e]) << 24);
                return radmath::bitsToFloat(bits);
            }
            case RadPackedEncoding::F16: {
                std::uint16_t h;
                memcpy(&h, plane + static_cast<std::size_t>(e) * 2u, 2u);
                return radmath::halfToFloat(h);
            }
            case RadPackedEncoding::F16LeBytes: {
                const std::uint32_t stride = count * dims;
                const std::uint16_t h = static_cast<std::uint16_t>(
                    plane[e] | (static_cast<std::uint16_t>(plane[stride + e]) << 8));
                return radmath::halfToFloat(h);
            }
            case RadPackedEncoding::R8:
                return radmath::dequantR8(plane[e], prop.min_val, prop.max_val - prop.min_val);
            case RadPackedEncoding::S8:
                return radmath::dequantS8(static_cast<std::int8_t>(plane[e]),
                                          radmath::shMaxAbs(prop.min_val, prop.max_val, prop.scale));
            case RadPackedEncoding::Ln0R8:
                return radmath::dequantLn0R8(plane[e], prop.min_val, prop.max_val);
            case RadPackedEncoding::LnF16: {
                std::uint16_t h;
                memcpy(&h, plane + static_cast<std::size_t>(e) * 2u, 2u);
                return std::exp(radmath::halfToFloat(h));
            }
            default:
                return 0.0f;
            }
        }

        struct PropSlots {
            const RadPackedProperty* by_kind[8];
        };

        __device__ float readShCanonical(const std::uint8_t* const slot,
                                         const PropSlots& props,
                                         const std::uint32_t count,
                                         const std::uint32_t i,
                                         const std::uint32_t c) {
            const std::uint32_t coeff = c / 3u;
            const std::uint32_t ch = c % 3u;
            const RadPackedProperty* prop = nullptr;
            std::uint32_t local = 0;
            std::uint32_t dims = 0;
            if (coeff < 3u) {
                prop = props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh1)];
                local = coeff;
                dims = 9u;
            } else if (coeff < 8u) {
                prop = props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh2)];
                local = coeff - 3u;
                dims = 15u;
            } else {
                prop = props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh3)];
                local = coeff - 8u;
                dims = 21u;
            }
            if (prop == nullptr) {
                return 0.0f;
            }
            return readPlane(slot + prop->plane_offset, *prop, dims, count, local * 3u + ch, i);
        }

        __global__ void lodPageDequantKernel(const std::uint8_t* const __restrict__ slot,
                                             const RadPagePackedDesc desc,
                                             const LodPoolDeviceView pool,
                                             const std::uint32_t page,
                                             const std::uint32_t page_splats) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= page_splats) {
                return;
            }
            const std::size_t dst = static_cast<std::size_t>(page) * page_splats + i;
            const bool live = i < desc.count;

            PropSlots props{};
            for (std::uint32_t p = 0; p < desc.property_count; ++p) {
                props.by_kind[desc.props[p].kind] = &desc.props[p];
            }
            const auto plane_of = [&](const RadPackedKind kind) {
                return props.by_kind[static_cast<std::uint32_t>(kind)];
            };

            if (const auto* const prop = plane_of(RadPackedKind::Means); pool.means != nullptr) {
                for (std::uint32_t d = 0; d < 3u; ++d) {
                    pool.means[dst * 3u + d] =
                        live && prop != nullptr
                            ? readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i)
                            : 0.0f;
                }
            }
            if (const auto* const prop = plane_of(RadPackedKind::Sh0); pool.sh0 != nullptr) {
                for (std::uint32_t d = 0; d < 3u; ++d) {
                    const float v = live && prop != nullptr
                                        ? readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i)
                                        : 0.5f;
                    pool.sh0[dst * 3u + d] = radmath::sh0Transform(v);
                }
            }
            if (const auto* const prop = plane_of(RadPackedKind::Scales); pool.scaling != nullptr) {
                for (std::uint32_t d = 0; d < 3u; ++d) {
                    const float v = live && prop != nullptr
                                        ? readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i)
                                        : 0.0f;
                    pool.scaling[dst * 3u + d] = radmath::scaleLog(v);
                }
            }
            if (const auto* const prop = plane_of(RadPackedKind::Alpha); pool.opacity != nullptr) {
                const float v = live && prop != nullptr
                                    ? readPlane(slot + prop->plane_offset, *prop, 1u, desc.count, 0u, i)
                                    : 0.0f;
                pool.opacity[dst] = desc.lod_opacity != 0u ? radmath::opacityLodEncoded(v)
                                                           : radmath::opacityLogit(v);
            }
            if (const auto* const prop = plane_of(RadPackedKind::Rotation); pool.rotation != nullptr) {
                float xyzw[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                if (live && prop != nullptr) {
                    if (static_cast<RadPackedEncoding>(prop->encoding) == RadPackedEncoding::Oct88R8) {
                        const std::uint8_t* const q = slot + prop->plane_offset +
                                                      static_cast<std::size_t>(i) * 3u;
                        radmath::dequantQuatOct88R8(q[0], q[1], q[2], xyzw);
                    } else {
                        for (std::uint32_t d = 0; d < 3u; ++d) {
                            xyzw[d] = readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i);
                        }
                        xyzw[3] = radmath::quatWFromXyz(xyzw[0], xyzw[1], xyzw[2]);
                    }
                }
                // Pool order is (w, x, y, z), as in decode_chunk_properties.
                pool.rotation[dst * 4u + 0u] = xyzw[3];
                pool.rotation[dst * 4u + 1u] = xyzw[0];
                pool.rotation[dst * 4u + 2u] = xyzw[1];
                pool.rotation[dst * 4u + 3u] = xyzw[2];
            }
            if (pool.shN != nullptr && pool.dst_slots > 0u) {
                // writeSwizzledShPage semantics: destination component k holds
                // canonical source float k while k < src_rest*3, else zero.
                const std::uint32_t active_floats = desc.sh_coeffs_rest * 3u;
                constexpr std::uint32_t kReorder = lfs::core::kShReorderSize;
                const std::uint32_t block = i / kReorder;
                const std::uint32_t lane = i % kReorder;
                const std::size_t page_blocks_base =
                    (static_cast<std::size_t>(page) * page_splats / kReorder + block) *
                    pool.dst_slots * kReorder;
                for (std::uint32_t s = 0; s < pool.dst_slots; ++s) {
                    float4 v{0.0f, 0.0f, 0.0f, 0.0f};
                    if (live) {
                        const std::uint32_t c0 = s * 4u;
                        v.x = c0 + 0u < active_floats ? readShCanonical(slot, props, desc.count, i, c0 + 0u) : 0.0f;
                        v.y = c0 + 1u < active_floats ? readShCanonical(slot, props, desc.count, i, c0 + 1u) : 0.0f;
                        v.z = c0 + 2u < active_floats ? readShCanonical(slot, props, desc.count, i, c0 + 2u) : 0.0f;
                        v.w = c0 + 3u < active_floats ? readShCanonical(slot, props, desc.count, i, c0 + 3u) : 0.0f;
                    }
                    const std::size_t float4_index =
                        page_blocks_base + static_cast<std::size_t>(s) * kReorder + lane;
                    reinterpret_cast<float4*>(pool.shN)[float4_index] = v;
                }
            }

            if (pool.meta_bounds != nullptr && pool.meta_links != nullptr) {
                if (i < desc.meta_node_count) {
                    constexpr float kInv = 1.0f / 65535.0f;
                    const std::uint8_t* const bq =
                        slot + desc.meta_bounds_offset + static_cast<std::size_t>(i) * 8u;
                    ushort4 q;
                    memcpy(&q, bq, 8u);
                    float4 bounds;
                    bounds.x = desc.frame.bbox_min[0] +
                               static_cast<float>(q.x) * kInv * desc.frame.bbox_extent[0];
                    bounds.y = desc.frame.bbox_min[1] +
                               static_cast<float>(q.y) * kInv * desc.frame.bbox_extent[1];
                    bounds.z = desc.frame.bbox_min[2] +
                               static_cast<float>(q.z) * kInv * desc.frame.bbox_extent[2];
                    bounds.w = std::exp(desc.frame.log_size_min +
                                        static_cast<float>(q.w) * kInv * desc.frame.log_size_range);
                    pool.meta_bounds[dst] = bounds;

                    const std::uint8_t* const lq =
                        slot + desc.meta_links_offset + static_cast<std::size_t>(i) * 12u;
                    uint4 links;
                    memcpy(&links, lq, 12u);
                    links.w = desc.chunk * page_splats + i;
                    pool.meta_links[dst] = links;
                } else {
                    pool.meta_bounds[dst] = float4{0.0f, 0.0f, 0.0f, 0.0f};
                    pool.meta_links[dst] = uint4{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
                }
            }
        }

    } // namespace

    cudaError_t launchLodPageDequant(const std::uint8_t* const device_slot,
                                     const lfs::io::RadPagePackedDesc& desc,
                                     const LodPoolDeviceView& pool,
                                     const std::uint32_t page,
                                     const std::uint32_t page_splats,
                                     const cudaStream_t stream) {
        constexpr std::uint32_t kBlock = 256;
        const std::uint32_t grid = (page_splats + kBlock - 1) / kBlock;
        lodPageDequantKernel<<<grid, kBlock, 0, stream>>>(device_slot, desc, pool, page, page_splats);
        return cudaGetLastError();
    }

} // namespace lfs::vis
