/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_render_directional_background_lobes(
        const float* lobe_dirs,
        const float* lobe_logits,
        float* output,
        int lobe_count,
        float lobe_sharpness,
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

    void launch_accumulate_directional_background_lobe_grad(
        const float* lobe_dirs,
        const float* lobe_logits,
        const float* grad_image,
        const float* alpha,
        const float* sky_gate,
        float* grad_lobes,
        int lobe_count,
        float lobe_sharpness,
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

    void launch_attenuate_sky_foreground_rgb_gradient(
        float* grad_image,
        bool grad_is_chw,
        const float* sky_gate,
        int height,
        int width,
        float strength,
        cudaStream_t stream = nullptr);

    void launch_attenuate_sky_foreground_error_map(
        float* error_map,
        const float* sky_gate,
        int height,
        int width,
        float strength,
        cudaStream_t stream = nullptr);

    void launch_directional_background_lobe_smoothness_grad(
        const float* lobe_dirs,
        const float* lobe_logits,
        float* grad_lobes,
        int lobe_count,
        float weight,
        cudaStream_t stream = nullptr);

    void launch_constrain_learned_sky_shell(
        float* means,
        float* scaling,
        float* rotation,
        float* opacity,
        float* sh0,
        float* shN,
        unsigned char* deleted,
        const float* shell_means,
        const float* shell_scaling,
        const float* shell_rotation,
        const float* shell_sh0_prior,
        int start,
        int count,
        int sh_rest,
        float opacity_min,
        float opacity_max,
        float sh0_min,
        float sh0_max,
        float prior_mix,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
