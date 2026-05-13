/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstdint>

namespace lfs::training {

    // Tiny per-pixel MLP that learns "is this pixel sky?" from
    // [R, G, B, saturation, luminance, rendered_alpha, y_norm, chroma_prior].
    // Operates as a drop-in replacement for the chroma-only sky gate once
    // bootstrapped. Pseudo-labels are derived from rendered alpha + the chroma
    // gate output; opaque (alpha > 0.88) pixels are labelled "not sky" and
    // empty + chroma-confident pixels are labelled "sky". Everything else is
    // ignored. The bootstrap phase keeps the chroma gate fully in charge so
    // the trainer never has to deal with a randomly-initialized discriminator.
    class SkyDiscriminator {
    public:
        // steps_scaler proportionally stretches every schedule threshold
        // (bootstrap, warmup duration, training cutoff). Values <= 0 are
        // treated as 1.0.
        void initialize(float steps_scaler = 1.0f);
        [[nodiscard]] bool is_initialized() const noexcept;
        [[nodiscard]] float steps_scaler() const noexcept { return steps_scaler_; }

        // True once the discriminator participates in the sky gate.
        // Schedule:
        //   iter <  bootstrap_iters:    chroma gate untouched, MLP not used
        //   bootstrap_iters .. warmup:  MLP trains, blends with chroma
        //   warmup .. train_cutoff:     MLP only, still training
        //   iter >= train_cutoff:       MLP frozen — forward only, no updates
        [[nodiscard]] bool is_active(int iter) const noexcept;
        [[nodiscard]] bool is_training_phase(int iter) const noexcept;

        // Linear chroma->mlp handover weight in [0, 1]. 0 = full MLP, 1 = full
        // chroma. Smooth ramp over the warmup window after bootstrap.
        [[nodiscard]] float chroma_blend(int iter) const noexcept;

        // Forward pass: overwrites sky_gate in place with the blended gate.
        void forward(
            const lfs::core::Tensor& target_image,
            bool target_is_chw,
            const lfs::core::Tensor& rendered_alpha,
            const lfs::core::Tensor& chroma_prior,
            int tile_y_offset,
            int full_height,
            lfs::core::Tensor& sky_gate,
            int iter);

        // Pseudo-labels + backward + Adam update. Returns false if there were
        // no usable pseudo-labels in this tile (in which case Adam is skipped).
        bool train_step(
            const lfs::core::Tensor& target_image,
            bool target_is_chw,
            const lfs::core::Tensor& rendered_alpha,
            const lfs::core::Tensor& chroma_prior,
            int tile_y_offset,
            int full_height,
            int iter);

        [[nodiscard]] int64_t step() const noexcept { return step_; }

    private:
        void zero_grad();

        lfs::core::Tensor weights_;
        lfs::core::Tensor grad_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor label_buffer_;
        lfs::core::Tensor sample_weight_buffer_;
        int64_t step_ = 0;
        int label_buffer_height_ = 0;
        int label_buffer_width_ = 0;
        bool frozen_logged_ = false;
        float steps_scaler_ = 1.0f;

        [[nodiscard]] int scaled(int v) const noexcept;

        static constexpr int kBootstrapIters = 1000;
        static constexpr int kWarmupIters = 1500;
        // Training cap: after this iter the MLP is frozen and only the forward
        // pass runs. By 5k iters the foreground has had 4k iters of pseudo-label
        // supervision plus 2.5k iters of full-strength MLP training, which is
        // more than enough to nail down the sky boundary; continuing to train
        // just burns cycles and risks late-stage drift.
        static constexpr int kTrainCutoffIters = 5000;
        static constexpr float kLearningRate = 5.0e-3f;
        static constexpr float kBeta1 = 0.9f;
        static constexpr float kBeta2 = 0.999f;
        static constexpr float kEps = 1.0e-8f;
    };

} // namespace lfs::training
