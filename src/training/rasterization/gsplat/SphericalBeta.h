/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Spherical-Beta view-dependent color, ported from the Beta-Splatting reference
// (Liu et al.). This augments the SH base color with `num_lobes` anisotropic lobes:
//
//   C = C_base + sum_i softplus(rgb_i) * max(dot(v, dir_i), 0)^(4 * exp(beta_i))
//
// where each lobe packs raw params [r, g, b, theta, phi, beta] (6 floats) and the
// lobe direction is (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
//
// The base color is taken as the input `colors` buffer (already SH-evaluated and
// offset by +0.5), so this is purely additive and orthogonal to the SH degree.

namespace gsplat_lfs {

    // Adds the spherical-beta lobe contribution in-place to `colors` ([N, 3]).
    void launch_spherical_beta_fwd_kernel(
        uint32_t num_lobes,
        const float* dirs,      // [N, 3] view directions (mean - campos)
        const float* sb_params, // [N, num_lobes, 6] raw lobe params
        const bool* masks,      // [N] optional
        int64_t total_elements, // N (= C * N_gauss)
        float* colors,          // [N, 3] in/out (base color in, += lobes)
        cudaStream_t stream = nullptr);

    // Backward: accumulates gradient w.r.t. raw sb_params. The gradient w.r.t. the base
    // color is the identity (dC/dC_base = 1), so v_colors is passed through unchanged to
    // the SH backward pass and does not need modification here.
    void launch_spherical_beta_bwd_kernel(
        uint32_t num_lobes,
        const float* dirs,      // [N, 3]
        const float* sb_params, // [N, num_lobes, 6]
        const bool* masks,      // [N] optional
        const float* v_colors,  // [N, 3] upstream gradient
        int64_t total_elements, // N
        float* v_sb_params,     // [N, num_lobes, 6] output gradient
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
