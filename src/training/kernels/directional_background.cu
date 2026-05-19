/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "directional_background.hpp"

#include <cmath>

namespace lfs::training::kernels {

    namespace {
        constexpr int kThreadsPerBlock = 256;
        constexpr int kTopLobes = 8;
        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kTwoPi = 6.28318530717958647692f;

        enum CameraModel {
            PINHOLE = 0,
            ORTHO = 1,
            FISHEYE = 2,
            EQUIRECTANGULAR = 3,
            THIN_PRISM_FISHEYE = 4
        };

        [[nodiscard]] inline unsigned int num_blocks_1d(const int total) {
            return static_cast<unsigned int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
        }

        __device__ inline void normalize3(float& x, float& y, float& z) {
            const float inv_len = rsqrtf(fmaxf(x * x + y * y + z * z, 1e-20f));
            x *= inv_len;
            y *= inv_len;
            z *= inv_len;
        }

        __device__ inline void camera_direction(
            const int camera_model,
            const float px_full,
            const float py_full,
            const int full_width,
            const int full_height,
            const float focal_x,
            const float focal_y,
            const float center_x,
            const float center_y,
            float& x,
            float& y,
            float& z) {
            if (camera_model == EQUIRECTANGULAR) {
                const float azimuth = kTwoPi * (px_full / static_cast<float>(full_width) - 0.5f);
                const float elevation = kPi * (py_full / static_cast<float>(full_height) - 0.5f);
                const float cos_elevation = cosf(elevation);
                x = cos_elevation * sinf(azimuth);
                y = sinf(elevation);
                z = cosf(azimuth) * cos_elevation;
            } else if (camera_model == ORTHO) {
                x = 0.0f;
                y = 0.0f;
                z = 1.0f;
            } else if (camera_model == FISHEYE || camera_model == THIN_PRISM_FISHEYE) {
                const float ux = (px_full - center_x) / focal_x;
                const float uy = (py_full - center_y) / focal_y;
                const float theta = sqrtf(ux * ux + uy * uy);
                if (theta > 1e-6f) {
                    const float sin_theta_over_theta = sinf(theta) / theta;
                    x = ux * sin_theta_over_theta;
                    y = uy * sin_theta_over_theta;
                    z = cosf(theta);
                } else {
                    x = 0.0f;
                    y = 0.0f;
                    z = 1.0f;
                }
            } else {
                x = (px_full - center_x) / focal_x;
                y = (py_full - center_y) / focal_y;
                z = 1.0f;
                normalize3(x, y, z);
            }
        }

        __device__ inline void world_direction(
            const float* __restrict__ world_view_transform,
            float& x,
            float& y,
            float& z) {
            const float cx = x;
            const float cy = y;
            const float cz = z;

            // world_view_transform is row-major world-to-camera. Directions transform by R^T.
            x = world_view_transform[0] * cx + world_view_transform[4] * cy + world_view_transform[8] * cz;
            y = world_view_transform[1] * cx + world_view_transform[5] * cy + world_view_transform[9] * cz;
            z = world_view_transform[2] * cx + world_view_transform[6] * cy + world_view_transform[10] * cz;
            normalize3(x, y, z);
        }

        __device__ inline float sigmoid_clamped(const float x) {
            const float clamped = fminf(fmaxf(x, -16.0f), 16.0f);
            return 1.0f / (1.0f + expf(-clamped));
        }

        __device__ inline float logit_clamped(const float x) {
            const float clamped = fminf(fmaxf(x, 1.0e-4f), 1.0f - 1.0e-4f);
            return logf(clamped / (1.0f - clamped));
        }

        __device__ inline float saturate(const float x) {
            return fminf(fmaxf(x, 0.0f), 1.0f);
        }

        __device__ inline float smoothstep01(const float edge0, const float edge1, const float x) {
            const float t = saturate((x - edge0) / fmaxf(edge1 - edge0, 1.0e-6f));
            return t * t * (3.0f - 2.0f * t);
        }

        __device__ inline float sky_prefix_color_confidence(const float r, const float g, const float b) {
            const float rr = saturate(r);
            const float gg = saturate(g);
            const float bb = saturate(b);
            const float maxc = fmaxf(rr, fmaxf(gg, bb));
            const float minc = fminf(rr, fminf(gg, bb));
            const float sat = maxc - minc;
            const float luma = 0.2126f * rr + 0.7152f * gg + 0.0722f * bb;
            if (luma < 0.08f || maxc < 0.12f) {
                return 0.0f;
            }

            const float warm_bias = fmaxf(rr - bb, gg - bb);
            const float cool_or_neutral = 1.0f - smoothstep01(0.04f, 0.18f, warm_bias);
            const float bright_neutral =
                smoothstep01(0.40f, 0.70f, luma) *
                (1.0f - smoothstep01(0.12f, 0.36f, sat)) *
                cool_or_neutral;
            const float blue =
                smoothstep01(0.02f, 0.18f, bb - rr) *
                smoothstep01(0.02f, 0.16f, bb - gg) *
                smoothstep01(0.34f, 0.62f, bb);
            const float green_reject =
                smoothstep01(0.02f, 0.18f, gg - fmaxf(rr, bb)) *
                smoothstep01(0.20f, 0.45f, gg);
            const float dark_reject = 1.0f - smoothstep01(0.10f, 0.28f, luma);
            return saturate(fmaxf(bright_neutral, blue) * (1.0f - green_reject) * (1.0f - dark_reject));
        }

        __device__ inline void fallback_sky_rgb(const float y, const float* __restrict__ fallback_color, float& r, float& g, float& b) {
            const float elevation = powf(saturate(-y), 0.55f);
            const float gradient_r = 0.82f * (1.0f - elevation) + 0.45f * elevation;
            const float gradient_g = 0.88f * (1.0f - elevation) + 0.62f * elevation;
            const float gradient_b = 0.96f * (1.0f - elevation) + 0.90f * elevation;
            const float base_r = fallback_color ? saturate(fallback_color[0]) : gradient_r;
            const float base_g = fallback_color ? saturate(fallback_color[1]) : gradient_g;
            const float base_b = fallback_color ? saturate(fallback_color[2]) : gradient_b;
            r = 0.72f * base_r + 0.28f * gradient_r;
            g = 0.72f * base_g + 0.28f * gradient_g;
            b = 0.72f * base_b + 0.28f * gradient_b;
        }

        __device__ inline void sanitize_sky_rgb(
            const float x,
            const float y,
            const float z,
            const float* __restrict__ fallback_color,
            float& r,
            float& g,
            float& b) {
            (void)x;
            (void)z;
            float fallback_r, fallback_g, fallback_b;
            fallback_sky_rgb(y, fallback_color, fallback_r, fallback_g, fallback_b);
            const float confidence = sky_prefix_color_confidence(r, g, b);
            const float keep = smoothstep01(0.10f, 0.45f, confidence);
            r = fallback_r * (1.0f - keep) + saturate(r) * keep;
            g = fallback_g * (1.0f - keep) + saturate(g) * keep;
            b = fallback_b * (1.0f - keep) + saturate(b) * keep;
        }

        __device__ inline void insert_top_lobe(
            const int index,
            const float dot,
            int (&top_indices)[kTopLobes],
            float (&top_dots)[kTopLobes]) {
            if (dot <= top_dots[kTopLobes - 1]) {
                return;
            }
            int slot = kTopLobes - 1;
            while (slot > 0 && dot > top_dots[slot - 1]) {
                top_dots[slot] = top_dots[slot - 1];
                top_indices[slot] = top_indices[slot - 1];
                --slot;
            }
            top_dots[slot] = dot;
            top_indices[slot] = index;
        }

        __device__ inline void find_top_lobes(
            const float* __restrict__ lobe_dirs,
            const int lobe_count,
            const float x,
            const float y,
            const float z,
            int (&top_indices)[kTopLobes],
            float (&top_dots)[kTopLobes]) {
            for (int i = 0; i < kTopLobes; ++i) {
                top_indices[i] = -1;
                top_dots[i] = -1.0e20f;
            }
            for (int i = 0; i < lobe_count; ++i) {
                const int base = i * 3;
                const float dot =
                    x * lobe_dirs[base + 0] +
                    y * lobe_dirs[base + 1] +
                    z * lobe_dirs[base + 2];
                insert_top_lobe(i, dot, top_indices, top_dots);
            }
        }

        __device__ inline void evaluate_lobe_color(
            const float* __restrict__ lobe_logits,
            const int (&top_indices)[kTopLobes],
            const float (&top_dots)[kTopLobes],
            const float sharpness,
            float (&rgb)[3],
            float (&weights)[kTopLobes],
            float& weight_sum) {
            rgb[0] = 0.0f;
            rgb[1] = 0.0f;
            rgb[2] = 0.0f;
            weight_sum = 0.0f;
            const float max_dot = top_dots[0];
            for (int k = 0; k < kTopLobes; ++k) {
                weights[k] = 0.0f;
                const int lobe = top_indices[k];
                if (lobe < 0) {
                    continue;
                }
                const float weight = expf(sharpness * (top_dots[k] - max_dot));
                weights[k] = weight;
                weight_sum += weight;
                const int base = lobe * 3;
                rgb[0] += weight * sigmoid_clamped(lobe_logits[base + 0]);
                rgb[1] += weight * sigmoid_clamped(lobe_logits[base + 1]);
                rgb[2] += weight * sigmoid_clamped(lobe_logits[base + 2]);
            }
            const float inv_weight_sum = weight_sum > 1.0e-8f ? 1.0f / weight_sum : 1.0f;
            rgb[0] *= inv_weight_sum;
            rgb[1] *= inv_weight_sum;
            rgb[2] *= inv_weight_sum;
        }

        __device__ inline float read_rgb_pixel(
            const float* __restrict__ image,
            const bool is_chw,
            const int c,
            const int idx,
            const int HW) {
            return is_chw ? image[c * HW + idx] : image[idx * 3 + c];
        }

        __device__ inline float read_luma_pixel(
            const float* __restrict__ image,
            const bool is_chw,
            const int x,
            const int y,
            const int width,
            const int height) {
            const int sx = x < 0 ? 0 : (x >= width ? width - 1 : x);
            const int sy = y < 0 ? 0 : (y >= height ? height - 1 : y);
            const int HW = width * height;
            const int idx = sy * width + sx;
            const float r = saturate(read_rgb_pixel(image, is_chw, 0, idx, HW));
            const float g = saturate(read_rgb_pixel(image, is_chw, 1, idx, HW));
            const float b = saturate(read_rgb_pixel(image, is_chw, 2, idx, HW));
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        __global__ void compute_auto_sky_gate_kernel(
            const float* __restrict__ target_image,
            const bool target_is_chw,
            float* __restrict__ sky_gate,
            const int height,
            const int width,
            const int full_height,
            const int full_width,
            const int tile_x_offset,
            const int tile_y_offset,
            const float focal_x,
            const float focal_y,
            const float center_x,
            const float center_y,
            const float* __restrict__ world_view_transform,
            const int camera_model,
            const float threshold) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total)
                return;

            const int x_pix = idx % width;
            const int y_pix = idx / width;
            const float px_full = static_cast<float>(x_pix + tile_x_offset) + 0.5f;
            const float py_full = static_cast<float>(y_pix + tile_y_offset) + 0.5f;

            float x, y, z;
            camera_direction(camera_model, px_full, py_full, full_width, full_height,
                             focal_x, focal_y, center_x, center_y, x, y, z);
            world_direction(world_view_transform, x, y, z);

            const int HW = total;
            const float r = saturate(read_rgb_pixel(target_image, target_is_chw, 0, idx, HW));
            const float g = saturate(read_rgb_pixel(target_image, target_is_chw, 1, idx, HW));
            const float b = saturate(read_rgb_pixel(target_image, target_is_chw, 2, idx, HW));
            const float maxc = fmaxf(r, fmaxf(g, b));
            const float minc = fminf(r, fminf(g, b));
            const float sat = maxc - minc;
            const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

            const float y_norm = py_full / fmaxf(static_cast<float>(full_height), 1.0f);
            const float top_prior = 1.0f - smoothstep01(0.42f, 0.86f, y_norm);
            const float placement = saturate(top_prior);

            const float warm_bias = fmaxf(r - b, g - b);
            const float cool_or_neutral = 1.0f - smoothstep01(0.04f, 0.18f, warm_bias);
            const float white_sky = smoothstep01(0.52f, 0.78f, luma) *
                                    (1.0f - smoothstep01(0.10f, 0.32f, sat)) *
                                    cool_or_neutral;
            const float blue_sky = smoothstep01(0.02f, 0.18f, b - r) *
                                   smoothstep01(0.02f, 0.16f, b - g) *
                                   smoothstep01(0.38f, 0.65f, b);
            const float green_reject = smoothstep01(0.02f, 0.18f, g - fmaxf(r, b)) *
                                       smoothstep01(0.20f, 0.45f, g);
            const float luma_edge = fmaxf(
                fabsf(luma - read_luma_pixel(target_image, target_is_chw, x_pix - 1, y_pix, width, height)),
                fmaxf(
                    fabsf(luma - read_luma_pixel(target_image, target_is_chw, x_pix + 1, y_pix, width, height)),
                    fmaxf(
                        fabsf(luma - read_luma_pixel(target_image, target_is_chw, x_pix, y_pix - 1, width, height)),
                        fabsf(luma - read_luma_pixel(target_image, target_is_chw, x_pix, y_pix + 1, width, height)))));
            const float smooth_region = 1.0f - smoothstep01(0.10f, 0.28f, luma_edge);
            const float color_confidence = fmaxf(white_sky, blue_sky) * (1.0f - green_reject) * smooth_region;
            const float confidence = placement * color_confidence;
            const float lo = fminf(fmaxf(threshold, 0.0f), 0.98f);
            const float hi = fminf(lo + 0.25f, 1.0f);
            sky_gate[idx] = smoothstep01(lo, hi, confidence);
        }

        __global__ void render_directional_background_lobes_kernel(
            const float* __restrict__ lobe_dirs,
            const float* __restrict__ lobe_logits,
            float* __restrict__ output,
            const int lobe_count,
            const float lobe_sharpness,
            const int height,
            const int width,
            const int full_height,
            const int full_width,
            const int tile_x_offset,
            const int tile_y_offset,
            const float focal_x,
            const float focal_y,
            const float center_x,
            const float center_y,
            const float* __restrict__ world_view_transform,
            const int camera_model) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total)
                return;

            const int x_pix = idx % width;
            const int y_pix = idx / width;
            const float px_full = static_cast<float>(x_pix + tile_x_offset) + 0.5f;
            const float py_full = static_cast<float>(y_pix + tile_y_offset) + 0.5f;

            float x, y, z;
            camera_direction(camera_model, px_full, py_full, full_width, full_height,
                             focal_x, focal_y, center_x, center_y, x, y, z);
            world_direction(world_view_transform, x, y, z);

            int top_indices[kTopLobes];
            float top_dots[kTopLobes];
            float weights[kTopLobes];
            float weight_sum = 0.0f;
            float rgb[3];
            find_top_lobes(lobe_dirs, lobe_count, x, y, z, top_indices, top_dots);
            evaluate_lobe_color(lobe_logits, top_indices, top_dots, lobe_sharpness, rgb, weights, weight_sum);

            const int HW = total;
            output[0 * HW + idx] = rgb[0];
            output[1 * HW + idx] = rgb[1];
            output[2 * HW + idx] = rgb[2];
        }

        __global__ void accumulate_directional_background_lobe_grad_kernel(
            const float* __restrict__ lobe_dirs,
            const float* __restrict__ lobe_logits,
            const float* __restrict__ grad_image,
            const float* __restrict__ alpha,
            const float* __restrict__ sky_gate,
            float* __restrict__ grad_lobes,
            const int lobe_count,
            const float lobe_sharpness,
            const int height,
            const int width,
            const int full_height,
            const int full_width,
            const int tile_x_offset,
            const int tile_y_offset,
            const float focal_x,
            const float focal_y,
            const float center_x,
            const float center_y,
            const float* __restrict__ world_view_transform,
            const int camera_model) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total) {
                return;
            }

            const int x_pix = idx % width;
            const int y_pix = idx / width;
            const float px_full = static_cast<float>(x_pix + tile_x_offset) + 0.5f;
            const float py_full = static_cast<float>(y_pix + tile_y_offset) + 0.5f;

            float x, y, z;
            camera_direction(camera_model, px_full, py_full, full_width, full_height,
                             focal_x, focal_y, center_x, center_y, x, y, z);
            world_direction(world_view_transform, x, y, z);

            const float gate = sky_gate ? saturate(sky_gate[idx]) : 1.0f;
            const float visibility = (1.0f - alpha[idx]) * gate;
            if (visibility <= 0.0f) {
                return;
            }

            int top_indices[kTopLobes];
            float top_dots[kTopLobes];
            float weights[kTopLobes];
            float weight_sum = 0.0f;
            float rgb[3];
            find_top_lobes(lobe_dirs, lobe_count, x, y, z, top_indices, top_dots);
            evaluate_lobe_color(lobe_logits, top_indices, top_dots, lobe_sharpness, rgb, weights, weight_sum);

            const int HW = total;
            const float inv_weight_sum = weight_sum > 1.0e-8f ? 1.0f / weight_sum : 1.0f;
            for (int k = 0; k < kTopLobes; ++k) {
                const int lobe = top_indices[k];
                if (lobe < 0) {
                    continue;
                }
                const float normalized_weight = weights[k] * inv_weight_sum;
                const int base = lobe * 3;
                for (int c = 0; c < 3; ++c) {
                    const float lobe_rgb = sigmoid_clamped(lobe_logits[base + c]);
                    const float grad_raw =
                        grad_image[c * HW + idx] *
                        visibility *
                        normalized_weight *
                        lobe_rgb *
                        (1.0f - lobe_rgb);
                    atomicAdd(&grad_lobes[base + c], grad_raw);
                }
            }
        }

        __global__ void attenuate_sky_foreground_rgb_gradient_kernel(
            float* __restrict__ grad_image,
            const bool grad_is_chw,
            const float* __restrict__ sky_gate,
            const int height,
            const int width,
            const float strength) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total) {
                return;
            }

            const float sky = sky_gate ? saturate(sky_gate[idx]) : 0.0f;
            const float clean_sky = smoothstep01(0.82f, 0.98f, sky);
            const float attenuation = saturate(strength) * clean_sky;
            const float factor = 1.0f - attenuation;
            const int HW = total;
            for (int c = 0; c < 3; ++c) {
                const int offset = grad_is_chw ? c * HW + idx : idx * 3 + c;
                grad_image[offset] *= factor;
            }
        }

        __global__ void attenuate_sky_foreground_error_map_kernel(
            float* __restrict__ error_map,
            const float* __restrict__ sky_gate,
            const int height,
            const int width,
            const float strength) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total) {
                return;
            }

            float sky = sky_gate ? saturate(sky_gate[idx]) : 0.0f;
            if (sky_gate) {
                // Dampen only very confident sky silhouettes; uncertain foliage/buildings still need growth.
                const int x = idx % width;
                const int y = idx / width;
                float neighbor_sky = sky;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= height) {
                        continue;
                    }
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = x + dx;
                        if (xx < 0 || xx >= width) {
                            continue;
                        }
                        neighbor_sky = fmaxf(neighbor_sky, saturate(sky_gate[yy * width + xx]));
                    }
                }
                sky = fmaxf(sky, 0.90f * smoothstep01(0.94f, 0.995f, neighbor_sky));
            }
            const float clean_sky = smoothstep01(0.82f, 0.98f, sky);
            const float attenuation = saturate(strength) * clean_sky;
            error_map[idx] *= 1.0f - attenuation;
        }

        __global__ void directional_background_lobe_smoothness_grad_kernel(
            const float* __restrict__ lobe_dirs,
            const float* __restrict__ lobe_logits,
            float* __restrict__ grad_lobes,
            const int lobe_count,
            const float weight) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = lobe_count * 3;
            if (idx >= total) {
                return;
            }

            const int lobe = idx / 3;
            const int channel = idx % 3;
            const int dir_base = lobe * 3;
            const float x = lobe_dirs[dir_base + 0];
            const float y = lobe_dirs[dir_base + 1];
            const float z = lobe_dirs[dir_base + 2];
            const float center = lobe_logits[idx];

            float neighbor_sum = 0.0f;
            float weight_sum = 0.0f;
            for (int other = 0; other < lobe_count; ++other) {
                if (other == lobe) {
                    continue;
                }
                const int other_dir_base = other * 3;
                const float dot =
                    x * lobe_dirs[other_dir_base + 0] +
                    y * lobe_dirs[other_dir_base + 1] +
                    z * lobe_dirs[other_dir_base + 2];
                const float w = expf(18.0f * (dot - 1.0f));
                neighbor_sum += w * lobe_logits[other * 3 + channel];
                weight_sum += w;
            }
            if (weight_sum <= 1.0e-8f) {
                return;
            }
            const float neighbor_mean = neighbor_sum / weight_sum;
            grad_lobes[idx] += weight * (center - neighbor_mean) / static_cast<float>(lobe_count);
        }

        __global__ void apply_directional_background_sky_prefix_prior_kernel(
            const float* __restrict__ lobe_dirs,
            const float* __restrict__ lobe_logits,
            const float* __restrict__ fallback_color,
            float* __restrict__ means,
            float* __restrict__ sh0,
            const float* __restrict__ center,
            const int lobe_count,
            const float lobe_sharpness,
            const int start,
            const int count,
            const float strength) {
            const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) {
                return;
            }

            const int row = start + i;
            float x = means[row * 3 + 0] - center[0];
            float y = means[row * 3 + 1] - center[1];
            float z = means[row * 3 + 2] - center[2];
            normalize3(x, y, z);

            int top_indices[kTopLobes];
            float top_dots[kTopLobes];
            float weights[kTopLobes];
            float weight_sum = 0.0f;
            float rgb[3];
            find_top_lobes(lobe_dirs, lobe_count, x, y, z, top_indices, top_dots);
            evaluate_lobe_color(lobe_logits, top_indices, top_dots, lobe_sharpness, rgb, weights, weight_sum);
            sanitize_sky_rgb(x, y, z, fallback_color, rgb[0], rgb[1], rgb[2]);

            constexpr float kSHC0 = 0.28209479177387814f;
            const float mix = saturate(strength);
            const float keep = 1.0f - mix;
            for (int c = 0; c < 3; ++c) {
                const float target_sh0 = (saturate(rgb[c]) - 0.5f) / kSHC0;
                sh0[row * 3 + c] = sh0[row * 3 + c] * keep + target_sh0 * mix;
            }
        }

        __global__ void accumulate_sky_prefix_lobe_stats_kernel(
            const float* __restrict__ lobe_dirs,
            const float* __restrict__ means,
            const float* __restrict__ sh0,
            const float* __restrict__ opacity,
            const float* __restrict__ scaling,
            const float* __restrict__ center,
            float* __restrict__ stats,
            float* __restrict__ weights,
            const int lobe_count,
            const float lobe_sharpness,
            const int start,
            const int count,
            const float min_opacity,
            const float max_log_scale) {
            const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) {
                return;
            }

            const int row = start + i;
            constexpr float kSHC0 = 0.28209479177387814f;
            const float rgb0 = saturate(sh0[row * 3 + 0] * kSHC0 + 0.5f);
            const float rgb1 = saturate(sh0[row * 3 + 1] * kSHC0 + 0.5f);
            const float rgb2 = saturate(sh0[row * 3 + 2] * kSHC0 + 0.5f);

            float confidence = sky_prefix_color_confidence(rgb0, rgb1, rgb2);
            if (confidence <= 0.02f) {
                return;
            }

            const float opa = sigmoid_clamped(opacity[row]);
            confidence *= 0.35f + 0.65f * smoothstep01(0.005f, fmaxf(min_opacity, 0.01f), opa);
            if (confidence <= 0.02f) {
                return;
            }

            float x = means[row * 3 + 0] - center[0];
            float y = means[row * 3 + 1] - center[1];
            float z = means[row * 3 + 2] - center[2];
            normalize3(x, y, z);

            const float log_scale =
                fminf((scaling[row * 3 + 0] + scaling[row * 3 + 1] + scaling[row * 3 + 2]) * (1.0f / 3.0f),
                      max_log_scale);
            const float target_opacity = fmaxf(opa, min_opacity);

            int top_indices[kTopLobes];
            float top_dots[kTopLobes];
            find_top_lobes(lobe_dirs, lobe_count, x, y, z, top_indices, top_dots);

            const float max_dot = top_dots[0];
            float weight_sum = 0.0f;
            float lobe_weights[kTopLobes];
            for (int k = 0; k < kTopLobes; ++k) {
                lobe_weights[k] = 0.0f;
                if (top_indices[k] < 0) {
                    continue;
                }
                const float w = expf(lobe_sharpness * (top_dots[k] - max_dot));
                lobe_weights[k] = w;
                weight_sum += w;
            }

            const float inv_weight_sum = weight_sum > 1.0e-8f ? 1.0f / weight_sum : 1.0f;
            for (int k = 0; k < kTopLobes; ++k) {
                const int lobe = top_indices[k];
                if (lobe < 0 || lobe_weights[k] <= 0.0f) {
                    continue;
                }
                const float w = confidence * lobe_weights[k] * inv_weight_sum;
                const int stat_base = lobe * 5;
                atomicAdd(&stats[stat_base + 0], w * rgb0);
                atomicAdd(&stats[stat_base + 1], w * rgb1);
                atomicAdd(&stats[stat_base + 2], w * rgb2);
                atomicAdd(&stats[stat_base + 3], w * target_opacity);
                atomicAdd(&stats[stat_base + 4], w * log_scale);
                atomicAdd(&weights[lobe], w);
            }
        }

        __global__ void apply_sky_prefix_lobe_stats_kernel(
            const float* __restrict__ lobe_dirs,
            const float* __restrict__ stats,
            const float* __restrict__ weights,
            const float* __restrict__ fallback_color,
            float* __restrict__ means,
            float* __restrict__ sh0,
            float* __restrict__ opacity,
            float* __restrict__ scaling,
            const float* __restrict__ center,
            const int lobe_count,
            const float propagation_sharpness,
            const int start,
            const int count,
            const float color_strength,
            const float opacity_strength,
            const float scale_strength,
            const float min_opacity,
            const float max_log_scale) {
            const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) {
                return;
            }

            const int row = start + i;
            float x = means[row * 3 + 0] - center[0];
            float y = means[row * 3 + 1] - center[1];
            float z = means[row * 3 + 2] - center[2];
            normalize3(x, y, z);

            for (int c = 0; c < 3; ++c) {
                const int scale_idx = row * 3 + c;
                scaling[scale_idx] = fminf(scaling[scale_idx], max_log_scale);
            }

            float rgb[3] = {0.0f, 0.0f, 0.0f};
            float target_opacity = 0.0f;
            float target_log_scale = 0.0f;
            float total_weight = 0.0f;

            for (int lobe = 0; lobe < lobe_count; ++lobe) {
                const float seed_weight = weights[lobe];
                if (seed_weight <= 1.0e-5f) {
                    continue;
                }
                const int dir_base = lobe * 3;
                const float dot =
                    x * lobe_dirs[dir_base + 0] +
                    y * lobe_dirs[dir_base + 1] +
                    z * lobe_dirs[dir_base + 2];
                const float directional_weight = expf(propagation_sharpness * (dot - 1.0f));
                const float w = directional_weight * sqrtf(seed_weight);
                if (w <= 1.0e-8f) {
                    continue;
                }

                const float inv_seed = 1.0f / seed_weight;
                const int stat_base = lobe * 5;
                rgb[0] += w * stats[stat_base + 0] * inv_seed;
                rgb[1] += w * stats[stat_base + 1] * inv_seed;
                rgb[2] += w * stats[stat_base + 2] * inv_seed;
                target_opacity += w * stats[stat_base + 3] * inv_seed;
                target_log_scale += w * stats[stat_base + 4] * inv_seed;
                total_weight += w;
            }

            if (total_weight <= 1.0e-8f) {
                return;
            }

            const float inv_total = 1.0f / total_weight;
            rgb[0] = saturate(rgb[0] * inv_total);
            rgb[1] = saturate(rgb[1] * inv_total);
            rgb[2] = saturate(rgb[2] * inv_total);
            sanitize_sky_rgb(x, y, z, fallback_color, rgb[0], rgb[1], rgb[2]);
            target_opacity = fmaxf(saturate(target_opacity * inv_total), min_opacity);
            target_log_scale = fminf(target_log_scale * inv_total, max_log_scale);

            constexpr float kSHC0 = 0.28209479177387814f;
            const float cur0 = saturate(sh0[row * 3 + 0] * kSHC0 + 0.5f);
            const float cur1 = saturate(sh0[row * 3 + 1] * kSHC0 + 0.5f);
            const float cur2 = saturate(sh0[row * 3 + 2] * kSHC0 + 0.5f);
            const float current_confidence = sky_prefix_color_confidence(cur0, cur1, cur2);
            const float color_mix = saturate(color_strength * (1.0f + 3.0f * (1.0f - current_confidence)));
            const float color_keep = 1.0f - color_mix;
            for (int c = 0; c < 3; ++c) {
                const float target_sh0 = (rgb[c] - 0.5f) / kSHC0;
                sh0[row * 3 + c] = sh0[row * 3 + c] * color_keep + target_sh0 * color_mix;
            }

            const float current_opacity = sigmoid_clamped(opacity[row]);
            target_opacity = fmaxf(target_opacity, current_opacity);
            const float opacity_floor_boost = current_opacity < min_opacity ? 0.25f : 0.0f;
            const float opacity_mix = saturate(opacity_strength + opacity_floor_boost);
            const float opacity_keep = 1.0f - opacity_mix;
            opacity[row] = opacity[row] * opacity_keep + logit_clamped(target_opacity) * opacity_mix;

            const float scale_mix = saturate(scale_strength);
            const float scale_keep = 1.0f - scale_mix;
            for (int c = 0; c < 3; ++c) {
                const int idx = row * 3 + c;
                const float current = fminf(scaling[idx], max_log_scale);
                const float target = fminf(fmaxf(current, target_log_scale), max_log_scale);
                scaling[idx] = current * scale_keep + target * scale_mix;
            }
        }

    } // namespace

    void launch_render_directional_background_lobes(
        const float* lobe_dirs,
        const float* lobe_logits,
        float* output,
        const int lobe_count,
        const float lobe_sharpness,
        const int height,
        const int width,
        const int full_height,
        const int full_width,
        const int tile_x_offset,
        const int tile_y_offset,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float* world_view_transform,
        const int camera_model,
        cudaStream_t stream) {
        const int total = height * width;
        render_directional_background_lobes_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            lobe_dirs, lobe_logits, output, lobe_count, lobe_sharpness, height, width, full_height, full_width,
            tile_x_offset, tile_y_offset, focal_x, focal_y, center_x, center_y, world_view_transform, camera_model);
    }

    void launch_accumulate_directional_background_lobe_grad(
        const float* lobe_dirs,
        const float* lobe_logits,
        const float* grad_image,
        const float* alpha,
        const float* sky_gate,
        float* grad_lobes,
        const int lobe_count,
        const float lobe_sharpness,
        const int height,
        const int width,
        const int full_height,
        const int full_width,
        const int tile_x_offset,
        const int tile_y_offset,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float* world_view_transform,
        const int camera_model,
        cudaStream_t stream) {
        const int total = height * width;
        accumulate_directional_background_lobe_grad_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            lobe_dirs, lobe_logits, grad_image, alpha, sky_gate, grad_lobes, lobe_count, lobe_sharpness, height, width,
            full_height, full_width, tile_x_offset, tile_y_offset, focal_x, focal_y, center_x, center_y,
            world_view_transform, camera_model);
    }

    void launch_compute_auto_sky_gate(
        const float* target_image,
        const bool target_is_chw,
        float* sky_gate,
        const int height,
        const int width,
        const int full_height,
        const int full_width,
        const int tile_x_offset,
        const int tile_y_offset,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float* world_view_transform,
        const int camera_model,
        const float threshold,
        cudaStream_t stream) {
        const int total = height * width;
        compute_auto_sky_gate_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            target_image,
            target_is_chw,
            sky_gate,
            height,
            width,
            full_height,
            full_width,
            tile_x_offset,
            tile_y_offset,
            focal_x,
            focal_y,
            center_x,
            center_y,
            world_view_transform,
            camera_model,
            threshold);
    }

    void launch_attenuate_sky_foreground_rgb_gradient(
        float* grad_image,
        const bool grad_is_chw,
        const float* sky_gate,
        const int height,
        const int width,
        const float strength,
        cudaStream_t stream) {
        const int total = height * width;
        attenuate_sky_foreground_rgb_gradient_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            grad_image, grad_is_chw, sky_gate, height, width, strength);
    }

    void launch_attenuate_sky_foreground_error_map(
        float* error_map,
        const float* sky_gate,
        const int height,
        const int width,
        const float strength,
        cudaStream_t stream) {
        const int total = height * width;
        attenuate_sky_foreground_error_map_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            error_map, sky_gate, height, width, strength);
    }

    void launch_directional_background_lobe_smoothness_grad(
        const float* lobe_dirs,
        const float* lobe_logits,
        float* grad_lobes,
        const int lobe_count,
        const float weight,
        cudaStream_t stream) {
        const int total = lobe_count * 3;
        directional_background_lobe_smoothness_grad_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            lobe_dirs, lobe_logits, grad_lobes, lobe_count, weight);
    }

    void launch_apply_directional_background_sky_prefix_prior(
        const float* lobe_dirs,
        const float* lobe_logits,
        const float* fallback_color,
        float* means,
        float* sh0,
        const float* center,
        const int lobe_count,
        const float lobe_sharpness,
        const int start,
        const int count,
        const float strength,
        cudaStream_t stream) {
        if (count <= 0) {
            return;
        }
        apply_directional_background_sky_prefix_prior_kernel<<<num_blocks_1d(count), kThreadsPerBlock, 0, stream>>>(
            lobe_dirs,
            lobe_logits,
            fallback_color,
            means,
            sh0,
            center,
            lobe_count,
            lobe_sharpness,
            start,
            count,
            strength);
    }

    void launch_accumulate_sky_prefix_lobe_stats(
        const float* lobe_dirs,
        const float* means,
        const float* sh0,
        const float* opacity,
        const float* scaling,
        const float* center,
        float* stats,
        float* weights,
        const int lobe_count,
        const float lobe_sharpness,
        const int start,
        const int count,
        const float min_opacity,
        const float max_log_scale,
        cudaStream_t stream) {
        if (count <= 0 || lobe_count <= 0) {
            return;
        }
        accumulate_sky_prefix_lobe_stats_kernel<<<num_blocks_1d(count), kThreadsPerBlock, 0, stream>>>(
            lobe_dirs,
            means,
            sh0,
            opacity,
            scaling,
            center,
            stats,
            weights,
            lobe_count,
            lobe_sharpness,
            start,
            count,
            min_opacity,
            max_log_scale);
    }

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
        const int lobe_count,
        const float propagation_sharpness,
        const int start,
        const int count,
        const float color_strength,
        const float opacity_strength,
        const float scale_strength,
        const float min_opacity,
        const float max_log_scale,
        cudaStream_t stream) {
        if (count <= 0 || lobe_count <= 0) {
            return;
        }
        apply_sky_prefix_lobe_stats_kernel<<<num_blocks_1d(count), kThreadsPerBlock, 0, stream>>>(
            lobe_dirs,
            stats,
            weights,
            fallback_color,
            means,
            sh0,
            opacity,
            scaling,
            center,
            lobe_count,
            propagation_sharpness,
            start,
            count,
            color_strength,
            opacity_strength,
            scale_strength,
            min_opacity,
            max_log_scale);
    }

} // namespace lfs::training::kernels
