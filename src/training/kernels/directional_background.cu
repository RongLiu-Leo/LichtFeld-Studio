/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "directional_background.hpp"

#include <cmath>

namespace lfs::training::kernels {

    namespace {
        constexpr int kThreadsPerBlock = 256;
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

        __host__ __device__ inline int sh_basis_count(const int degree) {
            return (degree + 1) * (degree + 1);
        }

        __device__ inline void eval_basis(const int degree, const float x, const float y, const float z, float* basis) {
            basis[0] = 1.0f;
            if (degree >= 1) {
                basis[1] = x;
                basis[2] = y;
                basis[3] = z;
            }
            if (degree >= 2) {
                basis[4] = x * y;
                basis[5] = y * z;
                basis[6] = z * x;
                basis[7] = x * x - y * y;
                basis[8] = 3.0f * z * z - 1.0f;
            }
        }

        __device__ inline float sigmoid_clamped(const float x) {
            const float clamped = fminf(fmaxf(x, -16.0f), 16.0f);
            return 1.0f / (1.0f + expf(-clamped));
        }

        __device__ inline float saturate(const float x) {
            return fminf(fmaxf(x, 0.0f), 1.0f);
        }

        __device__ inline float smoothstep01(const float edge0, const float edge1, const float x) {
            const float t = saturate((x - edge0) / fmaxf(edge1 - edge0, 1.0e-6f));
            return t * t * (3.0f - 2.0f * t);
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
            const float top_prior = 1.0f - smoothstep01(0.38f, 0.82f, y_norm);
            const float up_prior = smoothstep01(-0.10f, 0.35f, y);
            const float placement = saturate(fmaxf(top_prior, 0.75f * up_prior));

            const float white_sky = smoothstep01(0.55f, 0.82f, luma) *
                                    (1.0f - smoothstep01(0.10f, 0.34f, sat));
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

        __global__ void render_directional_background_sh_kernel(
            const float* __restrict__ coeffs,
            float* __restrict__ output,
            const int degree,
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

            float basis[9];
            eval_basis(degree, x, y, z, basis);
            const int basis_count = sh_basis_count(degree);

            float rgb[3] = {0.0f, 0.0f, 0.0f};
            for (int b = 0; b < basis_count; ++b) {
                const float sh = basis[b];
                rgb[0] += coeffs[b * 3 + 0] * sh;
                rgb[1] += coeffs[b * 3 + 1] * sh;
                rgb[2] += coeffs[b * 3 + 2] * sh;
            }

            const int HW = total;
            output[0 * HW + idx] = sigmoid_clamped(rgb[0]);
            output[1 * HW + idx] = sigmoid_clamped(rgb[1]);
            output[2 * HW + idx] = sigmoid_clamped(rgb[2]);
        }

        __global__ void accumulate_directional_background_sh_grad_kernel(
            const float* __restrict__ coeffs,
            const float* __restrict__ grad_image,
            const float* __restrict__ alpha,
            const float* __restrict__ sky_gate,
            float* __restrict__ grad_coeffs,
            const int degree,
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
            __shared__ float block_grad[27];
            if (threadIdx.x < 27) {
                block_grad[threadIdx.x] = 0.0f;
            }
            __syncthreads();

            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx < total) {
                const int x_pix = idx % width;
                const int y_pix = idx / width;
                const float px_full = static_cast<float>(x_pix + tile_x_offset) + 0.5f;
                const float py_full = static_cast<float>(y_pix + tile_y_offset) + 0.5f;

                float x, y, z;
                camera_direction(camera_model, px_full, py_full, full_width, full_height,
                                 focal_x, focal_y, center_x, center_y, x, y, z);
                world_direction(world_view_transform, x, y, z);

                float basis[9];
                eval_basis(degree, x, y, z, basis);
                const int basis_count = sh_basis_count(degree);
                const float gate = sky_gate ? saturate(sky_gate[idx]) : 1.0f;
                const float visibility = (1.0f - alpha[idx]) * gate;
                const int HW = total;

                float raw_rgb[3] = {0.0f, 0.0f, 0.0f};
                for (int b = 0; b < basis_count; ++b) {
                    const float sh = basis[b];
                    raw_rgb[0] += coeffs[b * 3 + 0] * sh;
                    raw_rgb[1] += coeffs[b * 3 + 1] * sh;
                    raw_rgb[2] += coeffs[b * 3 + 2] * sh;
                }
                const float rgb[3] = {
                    sigmoid_clamped(raw_rgb[0]),
                    sigmoid_clamped(raw_rgb[1]),
                    sigmoid_clamped(raw_rgb[2]),
                };
                const float grad_rgb[3] = {
                    grad_image[0 * HW + idx] * visibility * rgb[0] * (1.0f - rgb[0]),
                    grad_image[1 * HW + idx] * visibility * rgb[1] * (1.0f - rgb[1]),
                    grad_image[2 * HW + idx] * visibility * rgb[2] * (1.0f - rgb[2])};

                for (int b = 0; b < basis_count; ++b) {
                    const float sh = basis[b];
                    atomicAdd(&block_grad[b * 3 + 0], grad_rgb[0] * sh);
                    atomicAdd(&block_grad[b * 3 + 1], grad_rgb[1] * sh);
                    atomicAdd(&block_grad[b * 3 + 2], grad_rgb[2] * sh);
                }
            }
            __syncthreads();

            const int basis_count = sh_basis_count(degree);
            const int gradient_count = basis_count * 3;
            if (threadIdx.x < gradient_count) {
                atomicAdd(&grad_coeffs[threadIdx.x], block_grad[threadIdx.x]);
            }
        }

        __global__ void directional_background_l2_grad_kernel(
            const float* __restrict__ coeffs,
            float* __restrict__ grad_coeffs,
            const int num_coeffs,
            const float weight) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= num_coeffs)
                return;
            if (idx < 3)
                return;
            grad_coeffs[idx] += weight * coeffs[idx];
        }

    } // namespace

    void launch_render_directional_background_sh(
        const float* coeffs,
        float* output,
        const int degree,
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
        render_directional_background_sh_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            coeffs, output, degree, height, width, full_height, full_width, tile_x_offset, tile_y_offset,
            focal_x, focal_y, center_x, center_y, world_view_transform, camera_model);
    }

    void launch_accumulate_directional_background_sh_grad(
        const float* coeffs,
        const float* grad_image,
        const float* alpha,
        const float* sky_gate,
        float* grad_coeffs,
        const int degree,
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
        accumulate_directional_background_sh_grad_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            coeffs, grad_image, alpha, sky_gate, grad_coeffs, degree, height, width, full_height, full_width,
            tile_x_offset, tile_y_offset, focal_x, focal_y, center_x, center_y,
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

    void launch_directional_background_l2_grad(
        const float* coeffs,
        float* grad_coeffs,
        const int degree,
        const float weight,
        cudaStream_t stream) {
        const int num_coeffs = sh_basis_count(degree) * 3;
        directional_background_l2_grad_kernel<<<num_blocks_1d(num_coeffs), kThreadsPerBlock, 0, stream>>>(
            coeffs, grad_coeffs, num_coeffs, weight);
    }

} // namespace lfs::training::kernels
