/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <vector>

namespace lfs::training {

    struct DirectionalBackgroundSnapshot {
        int degree = 0;
        int texture_width = 0;
        int texture_height = 0;
        int lobe_count = 0;
        int64_t step = 0;
        // Row-major RGB display-linear equirectangular texture.
        std::vector<float> texture;
        // Row-major XYZ fixed sky-lobe directions and RGB display-linear lobe colors.
        std::vector<float> lobe_dirs;
        std::vector<float> lobe_colors;
        std::array<std::array<float, 3>, 9> coeffs{};
    };

    class DirectionalBackground {
    public:
        void initialize(const std::array<float, 3>& color, int degree, const std::array<float, 3>& up_axis = {0.0f, -1.0f, 0.0f});
        void ensure_initialized(const std::array<float, 3>& color, int degree, const std::array<float, 3>& up_axis = {0.0f, -1.0f, 0.0f});
        [[nodiscard]] const std::array<float, 3>& up_axis() const noexcept { return up_axis_; }

        [[nodiscard]] bool is_initialized() const noexcept { return lobe_logits_.is_valid() && lobe_logits_.numel() > 0; }
        [[nodiscard]] int degree() const noexcept { return degree_; }
        [[nodiscard]] int texture_width() const noexcept { return snapshot_texture_width_; }
        [[nodiscard]] int texture_height() const noexcept { return snapshot_texture_height_; }
        [[nodiscard]] int lobe_count() const noexcept { return lobe_count_; }
        [[nodiscard]] int64_t step() const noexcept { return step_; }
        [[nodiscard]] const lfs::core::Tensor& lobe_logits() const noexcept { return lobe_logits_; }
        [[nodiscard]] const lfs::core::Tensor& lobe_dirs() const noexcept { return lobe_dirs_; }
        [[nodiscard]] const lfs::core::Tensor& fallback_color() const noexcept { return fallback_color_; }
        [[nodiscard]] std::optional<DirectionalBackgroundSnapshot> snapshot() const;

        lfs::core::Tensor render(const lfs::core::Camera& camera,
                                 int width,
                                 int height,
                                 int tile_x_offset = 0,
                                 int tile_y_offset = 0,
                                 int full_width = 0,
                                 int full_height = 0);

        void accumulate_gradient(const lfs::core::Camera& camera,
                                 const lfs::core::Tensor& grad_image,
                                 const lfs::core::Tensor& alpha,
                                 const lfs::core::Tensor& sky_gate = {},
                                 int tile_x_offset = 0,
                                 int tile_y_offset = 0,
                                 int full_width = 0,
                                 int full_height = 0);

        void accumulate_target_gradient(const lfs::core::Camera& camera,
                                        const lfs::core::Tensor& target_image,
                                        const lfs::core::Tensor& sky_gate = {},
                                        int tile_x_offset = 0,
                                        int tile_y_offset = 0,
                                        int full_width = 0,
                                        int full_height = 0,
                                        float loss_scale = 1.0f);

        void apply_sky_prefix_prior(lfs::core::SplatData& model,
                                    const lfs::core::Tensor& center,
                                    int start,
                                    int count,
                                    float strength) const;

        void propagate_sky_prefix(lfs::core::SplatData& model,
                                  const lfs::core::Tensor& center,
                                  int start,
                                  int count,
                                  float color_strength,
                                  float opacity_strength,
                                  float scale_strength,
                                  float min_opacity,
                                  float max_scale);

        void add_l2_gradient(float weight);
        void optimizer_step(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
        void zero_grad();

        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

    private:
        lfs::core::Tensor lobe_logits_;
        lfs::core::Tensor lobe_dirs_;
        lfs::core::Tensor grad_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor render_buffer_;
        lfs::core::Tensor alpha_zero_buffer_;
        lfs::core::Tensor sky_prefix_stats_;
        lfs::core::Tensor sky_prefix_weights_;
        lfs::core::Tensor fallback_color_;
        std::array<float, 3> fallback_color_cpu_ = {0.82f, 0.88f, 0.96f};
        int degree_ = 2;
        int lobe_count_ = 0;
        int snapshot_texture_width_ = 0;
        int snapshot_texture_height_ = 0;
        int render_width_ = 0;
        int render_height_ = 0;
        int64_t step_ = 0;
        std::array<float, 3> up_axis_ = {0.0f, -1.0f, 0.0f};
    };

} // namespace lfs::training
