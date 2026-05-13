/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Per-pixel sky discriminator: a 2-layer MLP that maps per-pixel features
    // [R, G, B, saturation, luminance, rendered_alpha, y_norm, chroma_prior]
    // through a hidden ReLU layer of size kSkyDiscHiddenSize and a sigmoid
    // output. Weights live as a single packed buffer of size kSkyDiscParamCount
    // laid out as [W1 (F*H) | b1 (H) | W2 (H) | b2 (1)].

    inline constexpr int kSkyDiscFeatureDim = 8;
    inline constexpr int kSkyDiscHiddenSize = 32;
    inline constexpr int kSkyDiscParamCount =
        kSkyDiscFeatureDim * kSkyDiscHiddenSize +
        kSkyDiscHiddenSize +
        kSkyDiscHiddenSize +
        1;

    void launch_sky_discriminator_forward(
        const float* target_image,
        bool target_is_chw,
        const float* rendered_alpha,
        const float* chroma_prior,
        const float* weights,
        int height,
        int width,
        int full_height,
        int tile_y_offset,
        float chroma_blend,
        float* sky_gate_out,
        cudaStream_t stream = nullptr);

    void launch_sky_discriminator_pseudo_labels(
        const float* target_image,
        bool target_is_chw,
        const float* rendered_alpha,
        const float* chroma_prior,
        int height,
        int width,
        int full_height,
        int tile_y_offset,
        float* labels_out,
        float* sample_weights_out,
        cudaStream_t stream = nullptr);

    void launch_sky_discriminator_backward(
        const float* target_image,
        bool target_is_chw,
        const float* rendered_alpha,
        const float* chroma_prior,
        const float* weights,
        const float* labels,
        const float* sample_weights,
        int height,
        int width,
        int full_height,
        int tile_y_offset,
        float loss_scale,
        float* weight_grads,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
