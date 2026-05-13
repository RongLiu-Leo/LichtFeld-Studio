/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sky_discriminator.hpp"

#include <cmath>

namespace lfs::training::kernels {

    namespace {
        constexpr int kThreadsPerBlock = 256;
        constexpr int F = kSkyDiscFeatureDim;
        constexpr int Hdim = kSkyDiscHiddenSize;
        constexpr int kW1Offset = 0;
        constexpr int kB1Offset = kW1Offset + F * Hdim;
        constexpr int kW2Offset = kB1Offset + Hdim;
        constexpr int kB2Offset = kW2Offset + Hdim;

        [[nodiscard]] inline unsigned int num_blocks_1d(const int total) {
            return static_cast<unsigned int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
        }

        __device__ inline float saturate(const float x) {
            return fminf(fmaxf(x, 0.0f), 1.0f);
        }

        __device__ inline float read_rgb_pixel(
            const float* __restrict__ image,
            const bool is_chw,
            const int c,
            const int idx,
            const int HW) {
            return is_chw ? image[c * HW + idx] : image[idx * 3 + c];
        }

        __device__ inline void compute_features(
            const float* __restrict__ image,
            const bool is_chw,
            const float* __restrict__ alpha,
            const float* __restrict__ chroma_prior,
            const int idx,
            const int HW,
            const int x_pix,
            const int y_pix,
            const int tile_y_offset,
            const int full_height,
            float* __restrict__ features) {
            const float r = saturate(read_rgb_pixel(image, is_chw, 0, idx, HW));
            const float g = saturate(read_rgb_pixel(image, is_chw, 1, idx, HW));
            const float b = saturate(read_rgb_pixel(image, is_chw, 2, idx, HW));
            const float max_c = fmaxf(r, fmaxf(g, b));
            const float min_c = fminf(r, fminf(g, b));
            const float sat = max_c - min_c;
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const float a = alpha ? saturate(alpha[idx]) : 0.0f;
            const float y_norm = (static_cast<float>(y_pix + tile_y_offset) + 0.5f) /
                                 fmaxf(static_cast<float>(full_height), 1.0f);
            const float cprior = chroma_prior ? saturate(chroma_prior[idx]) : 0.0f;
            features[0] = r;
            features[1] = g;
            features[2] = b;
            features[3] = sat;
            features[4] = lum;
            features[5] = a;
            features[6] = y_norm;
            features[7] = cprior;
            (void)x_pix;
        }

        __device__ inline float sigmoid_clamped(const float x) {
            const float c = fminf(fmaxf(x, -16.0f), 16.0f);
            return 1.0f / (1.0f + expf(-c));
        }

        __device__ inline float mlp_forward(
            const float* __restrict__ features,
            const float* __restrict__ weights,
            float* __restrict__ hidden_out,  // [Hdim] - intermediate
            float* __restrict__ pre_relu_out // [Hdim] - z1 values, needed for backward
        ) {
            // Layer 1: z1 = W1 @ features + b1; h1 = relu(z1)
            const float* W1 = weights + kW1Offset;
            const float* b1 = weights + kB1Offset;
            for (int j = 0; j < Hdim; ++j) {
                float z = b1[j];
#pragma unroll
                for (int i = 0; i < F; ++i) {
                    z += W1[j * F + i] * features[i];
                }
                pre_relu_out[j] = z;
                hidden_out[j] = fmaxf(0.0f, z);
            }
            // Layer 2: z2 = W2 @ h1 + b2
            const float* W2 = weights + kW2Offset;
            const float* b2 = weights + kB2Offset;
            float z2 = b2[0];
            for (int j = 0; j < Hdim; ++j) {
                z2 += W2[j] * hidden_out[j];
            }
            return sigmoid_clamped(z2);
        }

        __global__ void sky_discriminator_forward_kernel(
            const float* __restrict__ target_image,
            const bool target_is_chw,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ chroma_prior,
            const float* __restrict__ weights,
            const int height,
            const int width,
            const int full_height,
            const int tile_y_offset,
            const float chroma_blend,
            float* __restrict__ sky_gate_out) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total) {
                return;
            }
            const int x_pix = idx % width;
            const int y_pix = idx / width;
            float features[F];
            compute_features(target_image, target_is_chw, rendered_alpha, chroma_prior,
                             idx, total, x_pix, y_pix, tile_y_offset, full_height, features);

            float hidden[Hdim];
            float pre_relu[Hdim];
            const float mlp_out = mlp_forward(features, weights, hidden, pre_relu);

            const float chroma_val = chroma_prior ? saturate(chroma_prior[idx]) : 0.0f;
            const float blend = saturate(chroma_blend);
            sky_gate_out[idx] = (1.0f - blend) * mlp_out + blend * chroma_val;
        }

        __global__ void sky_discriminator_pseudo_labels_kernel(
            const float* __restrict__ target_image,
            const bool target_is_chw,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ chroma_prior,
            const int height,
            const int width,
            const int full_height,
            const int tile_y_offset,
            float* __restrict__ labels_out,
            float* __restrict__ sample_weights_out) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total) {
                return;
            }
            const float r = saturate(read_rgb_pixel(target_image, target_is_chw, 0, idx, total));
            const float g = saturate(read_rgb_pixel(target_image, target_is_chw, 1, idx, total));
            const float b = saturate(read_rgb_pixel(target_image, target_is_chw, 2, idx, total));
            const float max_c = fmaxf(r, fmaxf(g, b));
            const float min_c = fminf(r, fminf(g, b));
            const float sat = max_c - min_c;
            const float a = rendered_alpha ? saturate(rendered_alpha[idx]) : 0.0f;
            const float cprior = chroma_prior ? saturate(chroma_prior[idx]) : 0.0f;
            const float y_norm = (static_cast<float>(idx / width + tile_y_offset) + 0.5f) /
                                 fmaxf(static_cast<float>(full_height), 1.0f);

            float label = 0.0f;
            float weight = 0.0f;

            // Strong negative: solidly opaque foreground.
            if (a > 0.88f) {
                label = 0.0f;
                weight = 1.0f;
            } else if (a > 0.70f && sat > 0.18f) {
                // Weak negative: opaque-ish + colorful surface.
                label = 0.0f;
                weight = 0.5f;
            }

            // Strong positive: empty pixel + chroma says sky + upper half of frame.
            if (a < 0.08f && cprior > 0.65f && y_norm < 0.75f) {
                label = 1.0f;
                weight = 1.0f;
            } else if (a < 0.04f && cprior > 0.85f) {
                // Very strong positive: nothing there + very high chroma confidence.
                label = 1.0f;
                weight = 1.0f;
            }

            labels_out[idx] = label;
            sample_weights_out[idx] = weight;
        }

        __global__ void sky_discriminator_backward_kernel(
            const float* __restrict__ target_image,
            const bool target_is_chw,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ chroma_prior,
            const float* __restrict__ weights,
            const float* __restrict__ labels,
            const float* __restrict__ sample_weights,
            const int height,
            const int width,
            const int full_height,
            const int tile_y_offset,
            const float loss_scale,
            float* __restrict__ weight_grads) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int total = height * width;
            if (idx >= total) {
                return;
            }
            const float sw = sample_weights[idx];
            if (sw <= 0.0f) {
                return;
            }

            const int x_pix = idx % width;
            const int y_pix = idx / width;

            float features[F];
            compute_features(target_image, target_is_chw, rendered_alpha, chroma_prior,
                             idx, total, x_pix, y_pix, tile_y_offset, full_height, features);

            float hidden[Hdim];
            float pre_relu[Hdim];
            const float p = mlp_forward(features, weights, hidden, pre_relu);

            // BCE loss gradient (sigmoid + BCE collapse): dz2 = (p - label) * weight * scale
            const float label = labels[idx];
            const float dz2 = (p - label) * sw * loss_scale;

            // Layer 2 grads: dW2[j] = dz2 * hidden[j], db2 = dz2
            const float* W2 = weights + kW2Offset;
            float dW2_g[Hdim];
            for (int j = 0; j < Hdim; ++j) {
                dW2_g[j] = dz2 * hidden[j];
            }
            // Propagate to hidden: dh1[j] = dz2 * W2[j], then through ReLU.
            float dz1[Hdim];
            for (int j = 0; j < Hdim; ++j) {
                const float dh1 = dz2 * W2[j];
                dz1[j] = pre_relu[j] > 0.0f ? dh1 : 0.0f;
            }

            // Layer 1 grads: dW1[j, i] = dz1[j] * features[i], db1[j] = dz1[j].
            for (int j = 0; j < Hdim; ++j) {
                const float dzj = dz1[j];
                if (dzj == 0.0f) {
                    continue;
                }
#pragma unroll
                for (int i = 0; i < F; ++i) {
                    atomicAdd(&weight_grads[kW1Offset + j * F + i], dzj * features[i]);
                }
                atomicAdd(&weight_grads[kB1Offset + j], dzj);
            }
            for (int j = 0; j < Hdim; ++j) {
                if (dW2_g[j] != 0.0f) {
                    atomicAdd(&weight_grads[kW2Offset + j], dW2_g[j]);
                }
            }
            atomicAdd(&weight_grads[kB2Offset], dz2);
        }

    } // namespace

    void launch_sky_discriminator_forward(
        const float* target_image,
        const bool target_is_chw,
        const float* rendered_alpha,
        const float* chroma_prior,
        const float* weights,
        const int height,
        const int width,
        const int full_height,
        const int tile_y_offset,
        const float chroma_blend,
        float* sky_gate_out,
        cudaStream_t stream) {
        const int total = height * width;
        sky_discriminator_forward_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            target_image, target_is_chw, rendered_alpha, chroma_prior, weights,
            height, width, full_height, tile_y_offset, chroma_blend, sky_gate_out);
    }

    void launch_sky_discriminator_pseudo_labels(
        const float* target_image,
        const bool target_is_chw,
        const float* rendered_alpha,
        const float* chroma_prior,
        const int height,
        const int width,
        const int full_height,
        const int tile_y_offset,
        float* labels_out,
        float* sample_weights_out,
        cudaStream_t stream) {
        const int total = height * width;
        sky_discriminator_pseudo_labels_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            target_image, target_is_chw, rendered_alpha, chroma_prior,
            height, width, full_height, tile_y_offset, labels_out, sample_weights_out);
    }

    void launch_sky_discriminator_backward(
        const float* target_image,
        const bool target_is_chw,
        const float* rendered_alpha,
        const float* chroma_prior,
        const float* weights,
        const float* labels,
        const float* sample_weights,
        const int height,
        const int width,
        const int full_height,
        const int tile_y_offset,
        const float loss_scale,
        float* weight_grads,
        cudaStream_t stream) {
        const int total = height * width;
        sky_discriminator_backward_kernel<<<num_blocks_1d(total), kThreadsPerBlock, 0, stream>>>(
            target_image, target_is_chw, rendered_alpha, chroma_prior, weights,
            labels, sample_weights,
            height, width, full_height, tile_y_offset, loss_scale, weight_grads);
    }

} // namespace lfs::training::kernels
