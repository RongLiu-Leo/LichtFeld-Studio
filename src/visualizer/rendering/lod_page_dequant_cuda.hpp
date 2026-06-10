/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/formats/rad_packed_page.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::vis {

    // Device-pointer view of the page-pool regions the dequant kernel writes.
    // Pointers are pre-offset region bases inside the CUDA-imported pool
    // buffers; indexing is by global pool splat (page * page_splats + i).
    struct LodPoolDeviceView {
        float* means = nullptr;    // [capacity*3]
        float* sh0 = nullptr;      // [capacity*3]
        float* shN = nullptr;      // swizzled float4 slot blocks
        float* rotation = nullptr; // [capacity*4], (w,x,y,z)
        float* scaling = nullptr;  // [capacity*3], log domain
        float* opacity = nullptr;  // [capacity]
        float4* meta_bounds = nullptr;
        uint4* meta_links = nullptr;
        std::uint32_t dst_rest = 0;
        std::uint32_t dst_slots = 0;
    };

    // Dequantizes one packed page (planes staged by decode_rad_chunk_packed,
    // already copied to `device_slot`) into the fp32 pool regions and expands
    // the sidecar bounds/links planes — GPU replacement for the CPU
    // decode + swizzle + expand_rad_meta_page path, launched on the upload
    // engine's stream between the slot copy and the timeline signal.
    cudaError_t launchLodPageDequant(const std::uint8_t* device_slot,
                                     const lfs::io::RadPagePackedDesc& desc,
                                     const LodPoolDeviceView& pool,
                                     std::uint32_t page,
                                     std::uint32_t page_splats,
                                     cudaStream_t stream);

} // namespace lfs::vis
