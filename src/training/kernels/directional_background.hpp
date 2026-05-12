/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_render_directional_background_sh(
        const float* coeffs,
        float* output,
        int degree,
        int height,
        int width,
        int full_height,
        int full_width,
        int tile_x_offset,
        int tile_y_offset,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        const float* world_view_transform,
        int camera_model,
        cudaStream_t stream = nullptr);

    void launch_accumulate_directional_background_sh_grad(
        const float* coeffs,
        const float* grad_image,
        const float* alpha,
        const float* sky_gate,
        float* grad_coeffs,
        int degree,
        int height,
        int width,
        int full_height,
        int full_width,
        int tile_x_offset,
        int tile_y_offset,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        const float* world_view_transform,
        int camera_model,
        cudaStream_t stream = nullptr);

    void launch_compute_auto_sky_gate(
        const float* target_image,
        bool target_is_chw,
        float* sky_gate,
        int height,
        int width,
        int full_height,
        int full_width,
        int tile_x_offset,
        int tile_y_offset,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        const float* world_view_transform,
        int camera_model,
        float threshold,
        cudaStream_t stream = nullptr);

    void launch_directional_background_l2_grad(
        const float* coeffs,
        float* grad_coeffs,
        int degree,
        float weight,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
