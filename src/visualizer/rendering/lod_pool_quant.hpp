/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Canonical quantized layout of the LOD page pool. Locked against the
// converter's streaming profile (means f32_lebytes, rgb r8_delta, scales
// ln_f16, orientation oct88r8, SH bands s8, alpha r8/f16) so the pool never
// adds loss over the file: s8 SH and log-f16 scales pass through bit-exact,
// f16 elsewhere is >=8x finer than the file's own 8-bit quantization.
// Shared by the dequant kernels (.cu), the upload engine, and the renderer's
// region layout; nvcc-safe.

#pragma once

#include <cstdint>

namespace lfs::vis::lodq {

    // Per-splat bytes in each pool region. xyz stays f32x3: position drives
    // culling/selection shaders that are not quant-aware, and the file stores
    // f32 there anyway.
    inline constexpr std::size_t kXyzBytes = 12;     // f32x3
    inline constexpr std::size_t kSh0Bytes = 8;      // f16x3 + 16-bit pad (uint2)
    inline constexpr std::size_t kShNSlotBytes = 4;  // s8x4 per float4 slot (uchar4)
    inline constexpr std::size_t kRotationBytes = 8; // f16x4, pool order (w,x,y,z)
    inline constexpr std::size_t kScalingBytes = 8;  // log-domain f16x3 + pad
    inline constexpr std::size_t kOpacityBytes = 2;  // f16, post lod/logit transform

    // Per-page dequant frame in the InputPageFrames region: float4[4].
    //   [0] = (sh1_max_abs, sh2_max_abs, sh3_max_abs, unused)
    //   [1..3] reserved (PR4: bbox/log-size frames for quantized node bounds)
    inline constexpr std::size_t kPageFrameBytes = 64;
    inline constexpr std::size_t kPageFrameFloat4s = kPageFrameBytes / 16;

    struct PageFrame {
        float sh_max[3] = {0.0f, 0.0f, 0.0f};
        float reserved0 = 0.0f;
        float reserved[12] = {};
    };
    static_assert(sizeof(PageFrame) == kPageFrameBytes);

} // namespace lfs::vis::lodq
