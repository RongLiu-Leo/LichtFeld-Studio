/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "SphericalBeta.h"

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    // Beta-Splatting reference uses softplus with beta = log(2) * 10 for the lobe color
    // channels so contributions stay non-negative. exp(beta_lobe) controls sharpness.
    constexpr float SB_SOFTPLUS_BETA = 6.931471805599453f; // log(2) * 10

    __device__ __forceinline__ float sb_softplus(const float x) {
        // (1/b) * log(1 + exp(b*x)), numerically stable for large b*x.
        const float bx = SB_SOFTPLUS_BETA * x;
        const float sp = (bx > 20.0f) ? bx : log1pf(__expf(bx));
        return sp / SB_SOFTPLUS_BETA;
    }

    __device__ __forceinline__ float sb_softplus_grad(const float x) {
        // d/dx softplus_b(x) = sigmoid(b*x)
        const float bx = SB_SOFTPLUS_BETA * x;
        return 1.0f / (1.0f + __expf(-bx));
    }

    __global__ void spherical_beta_fwd_kernel(
        const uint32_t N,
        const uint32_t num_lobes,
        const float* __restrict__ dirs,      // [N, 3]
        const float* __restrict__ sb_params, // [N, num_lobes, 6]
        const bool* __restrict__ masks,      // [N]
        float* __restrict__ colors) {        // [N, 3] in/out
        const uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }
        if (masks != nullptr && !masks[idx]) {
            return;
        }

        const float dx = dirs[idx * 3 + 0];
        const float dy = dirs[idx * 3 + 1];
        const float dz = dirs[idx * 3 + 2];
        const float inorm = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-20f);
        const float vx = dx * inorm, vy = dy * inorm, vz = dz * inorm;

        float cr = colors[idx * 3 + 0];
        float cg = colors[idx * 3 + 1];
        float cb = colors[idx * 3 + 2];

        const float* p = sb_params + static_cast<size_t>(idx) * num_lobes * 6u;
        for (uint32_t l = 0; l < num_lobes; ++l) {
            const float* lobe = p + l * 6u;
            const float r = sb_softplus(lobe[0]);
            const float g = sb_softplus(lobe[1]);
            const float b = sb_softplus(lobe[2]);
            const float theta = lobe[3];
            const float phi = lobe[4];
            const float beta = lobe[5];

            float st, ct, sp, cp;
            __sincosf(theta, &st, &ct);
            __sincosf(phi, &sp, &cp);
            const float lobe_x = st * cp;
            const float lobe_y = st * sp;
            const float lobe_z = ct;

            const float dot = vx * lobe_x + vy * lobe_y + vz * lobe_z;
            if (dot > 0.0f) {
                const float beta_term = __powf(dot, 4.0f * __expf(beta));
                cr += beta_term * r;
                cg += beta_term * g;
                cb += beta_term * b;
            }
        }

        colors[idx * 3 + 0] = cr;
        colors[idx * 3 + 1] = cg;
        colors[idx * 3 + 2] = cb;
    }

    __global__ void spherical_beta_bwd_kernel(
        const uint32_t N,
        const uint32_t num_lobes,
        const float* __restrict__ dirs,      // [N, 3]
        const float* __restrict__ sb_params, // [N, num_lobes, 6]
        const bool* __restrict__ masks,      // [N]
        const float* __restrict__ v_colors,  // [N, 3]
        float* __restrict__ v_sb_params) {   // [N, num_lobes, 6]
        const uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }
        float* vp_base = v_sb_params + static_cast<size_t>(idx) * num_lobes * 6u;
        if (masks != nullptr && !masks[idx]) {
            for (uint32_t i = 0; i < num_lobes * 6u; ++i)
                vp_base[i] = 0.0f;
            return;
        }

        const float dx = dirs[idx * 3 + 0];
        const float dy = dirs[idx * 3 + 1];
        const float dz = dirs[idx * 3 + 2];
        const float inorm = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-20f);
        const float vx = dx * inorm, vy = dy * inorm, vz = dz * inorm;

        const float vcr = v_colors[idx * 3 + 0];
        const float vcg = v_colors[idx * 3 + 1];
        const float vcb = v_colors[idx * 3 + 2];

        const float* p = sb_params + static_cast<size_t>(idx) * num_lobes * 6u;
        for (uint32_t l = 0; l < num_lobes; ++l) {
            const float* lobe = p + l * 6u;
            float* vlobe = vp_base + l * 6u;

            const float raw_r = lobe[0], raw_g = lobe[1], raw_b = lobe[2];
            const float r = sb_softplus(raw_r);
            const float g = sb_softplus(raw_g);
            const float b = sb_softplus(raw_b);
            const float theta = lobe[3];
            const float phi = lobe[4];
            const float beta = lobe[5];

            float st, ct, sp, cp;
            __sincosf(theta, &st, &ct);
            __sincosf(phi, &sp, &cp);
            const float lobe_x = st * cp;
            const float lobe_y = st * sp;
            const float lobe_z = ct;

            const float dot = vx * lobe_x + vy * lobe_y + vz * lobe_z;

            float beta_term = 0.0f;
            if (dot > 0.0f) {
                beta_term = __powf(dot, 4.0f * __expf(beta));
            }

            // Grad w.r.t. activated rgb, then chain through softplus to raw rgb.
            vlobe[0] = vcr * beta_term * sb_softplus_grad(raw_r);
            vlobe[1] = vcg * beta_term * sb_softplus_grad(raw_g);
            vlobe[2] = vcb * beta_term * sb_softplus_grad(raw_b);

            float grad_theta = 0.0f, grad_phi = 0.0f, grad_beta = 0.0f;
            if (dot > 0.0f) {
                const float exp_beta = __expf(beta);
                const float exponent = 4.0f * exp_beta;
                const float color_grad = vcr * r + vcg * g + vcb * b;

                const float d_dot_d_theta = vx * (ct * cp) + vy * (ct * sp) - vz * st;
                const float d_dot_d_phi = -vx * st * sp + vy * st * cp;

                const float pow_em1 = __powf(dot, exponent - 1.0f);
                const float d_bt_d_theta = exponent * pow_em1 * d_dot_d_theta;
                const float d_bt_d_phi = exponent * pow_em1 * d_dot_d_phi;
                const float d_bt_d_beta = exponent * beta_term * logf(dot);

                grad_theta = color_grad * d_bt_d_theta;
                grad_phi = color_grad * d_bt_d_phi;
                grad_beta = color_grad * d_bt_d_beta;
            }
            vlobe[3] = grad_theta;
            vlobe[4] = grad_phi;
            vlobe[5] = grad_beta;
        }
    }

    void launch_spherical_beta_fwd_kernel(
        uint32_t num_lobes,
        const float* dirs,
        const float* sb_params,
        const bool* masks,
        int64_t total_elements,
        float* colors,
        cudaStream_t stream) {
        const uint32_t N = static_cast<uint32_t>(total_elements);
        if (N == 0 || num_lobes == 0) {
            return;
        }
        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);
        spherical_beta_fwd_kernel<<<grid, threads, 0, stream>>>(
            N, num_lobes, dirs, sb_params, masks, colors);
    }

    void launch_spherical_beta_bwd_kernel(
        uint32_t num_lobes,
        const float* dirs,
        const float* sb_params,
        const bool* masks,
        const float* v_colors,
        int64_t total_elements,
        float* v_sb_params,
        cudaStream_t stream) {
        const uint32_t N = static_cast<uint32_t>(total_elements);
        if (N == 0 || num_lobes == 0) {
            return;
        }
        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);
        spherical_beta_bwd_kernel<<<grid, threads, 0, stream>>>(
            N, num_lobes, dirs, sb_params, masks, v_colors, v_sb_params);
    }

} // namespace gsplat_lfs
