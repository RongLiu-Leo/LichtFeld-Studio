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

    void launch_apply_directional_background_sky_prefix_prior(
        const float* lobe_dirs,
        const float* lobe_logits,
        const float* fallback_color,
        float* means,
        float* sh0,
        const float* center,
        int lobe_count,
        float lobe_sharpness,
        int start,
        int count,
        float strength,
        cudaStream_t stream = nullptr);

    void launch_accumulate_sky_prefix_lobe_stats(
        const float* lobe_dirs,
        const float* means,
        const float* sh0,
        const float* opacity,
        const float* scaling,
        const float* center,
        float* stats,
        float* weights,
        int lobe_count,
        float lobe_sharpness,
        int start,
        int count,
        float min_opacity,
        float max_log_scale,
        cudaStream_t stream = nullptr);

    void launch_apply_sky_prefix_lobe_stats(
        const float* lobe_dirs,
        const float* stats,
        const float* weights,
        const float* fallback_color,
        float* means,
        float* sh0,
        float* opacity,
        float* scaling,
        const float* center,
        int lobe_count,
        float propagation_sharpness,
        int start,
        int count,
        float color_strength,
        float opacity_strength,
        float scale_strength,
        float min_opacity,
        float max_log_scale,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
