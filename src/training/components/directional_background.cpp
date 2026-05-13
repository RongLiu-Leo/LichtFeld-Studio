/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "directional_background.hpp"

#include "adam_api.h"
#include "core/logger.hpp"
#include "training/kernels/directional_background.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr uint32_t DIRECTIONAL_BG_MAGIC = 0x4C464244; // "LFBD"
        constexpr uint32_t DIRECTIONAL_BG_VERSION = 4;
        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kTwoPi = 6.28318530717958647692f;

        int clamp_degree(const int degree) {
            return std::clamp(degree, 0, 2);
        }

        int lobe_count_for_degree(const int degree) {
            return 48 << clamp_degree(degree);
        }

        int snapshot_width_for_degree(const int degree) {
            return 64 << clamp_degree(degree);
        }

        int snapshot_height_for_degree(const int degree) {
            return snapshot_width_for_degree(degree) / 2;
        }

        float lobe_sharpness_for_degree(const int degree) {
            switch (clamp_degree(degree)) {
            case 0:
                return 18.0f;
            case 1:
                return 32.0f;
            default:
                return 56.0f;
            }
        }

        float clamp_unit_color(const float value) {
            return std::clamp(value, 0.02f, 0.98f);
        }

        float learned_initial_color(const float value) {
            return value <= 1.0e-4f ? 0.5f : value;
        }

        float logit_unit_color(const float value) {
            const float clamped = clamp_unit_color(value);
            return std::log(clamped / (1.0f - clamped));
        }

        float sigmoid_clamped(const float value) {
            const float clamped = std::clamp(value, -16.0f, 16.0f);
            return 1.0f / (1.0f + std::exp(-clamped));
        }

        float saturate(const float value) {
            return std::clamp(value, 0.0f, 1.0f);
        }

        lfs::core::Tensor contiguous_cuda(lfs::core::Tensor tensor) {
            if (tensor.device() != lfs::core::Device::CUDA) {
                tensor = tensor.cuda();
            }
            return tensor.is_contiguous() ? tensor : tensor.contiguous();
        }

        std::array<float, 3> normalize_or(const std::array<float, 3>& v, const std::array<float, 3>& fallback) {
            const float sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (sq <= 1.0e-12f) {
                return fallback;
            }
            const float inv = 1.0f / std::sqrt(sq);
            return {v[0] * inv, v[1] * inv, v[2] * inv};
        }

        std::array<float, 3> resolve_up_axis(const std::array<float, 3>& requested) {
            return normalize_or(requested, {0.0f, 1.0f, 0.0f});
        }

        std::vector<float> make_lobe_dirs(const int lobe_count, const std::array<float, 3>& up_axis) {
            std::vector<float> dirs(static_cast<size_t>(lobe_count) * 3UL);
            constexpr float min_y = -0.12f;
            constexpr float golden_angle = kPi * (3.0f - 2.2360679774997896964f);

            const auto up = resolve_up_axis(up_axis);
            const std::array<float, 3> reference =
                (std::abs(up[1]) < 0.85f) ? std::array<float, 3>{0.0f, 1.0f, 0.0f}
                                          : std::array<float, 3>{1.0f, 0.0f, 0.0f};
            const std::array<float, 3> right_raw{
                reference[1] * up[2] - reference[2] * up[1],
                reference[2] * up[0] - reference[0] * up[2],
                reference[0] * up[1] - reference[1] * up[0]};
            const auto right = normalize_or(right_raw, {1.0f, 0.0f, 0.0f});
            const std::array<float, 3> forward{
                up[1] * right[2] - up[2] * right[1],
                up[2] * right[0] - up[0] * right[2],
                up[0] * right[1] - up[1] * right[0]};

            for (int i = 0; i < lobe_count; ++i) {
                const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(lobe_count);
                const float y = 1.0f - t * (1.0f - min_y);
                const float radius = std::sqrt(std::max(0.0f, 1.0f - y * y));
                const float theta = static_cast<float>(i) * golden_angle;
                const float local_x = std::cos(theta) * radius;
                const float local_z = std::sin(theta) * radius;
                const size_t base = static_cast<size_t>(i) * 3UL;
                dirs[base + 0] = right[0] * local_x + up[0] * y + forward[0] * local_z;
                dirs[base + 1] = right[1] * local_x + up[1] * y + forward[1] * local_z;
                dirs[base + 2] = right[2] * local_x + up[2] * y + forward[2] * local_z;
            }
            return dirs;
        }

        std::array<float, 3> default_sky_color_for_dir(
            const std::array<float, 3>& base_color,
            const float y) {
            const std::array<float, 3> below{0.74f, 0.76f, 0.78f};
            const std::array<float, 3> horizon{0.86f, 0.91f, 0.98f};
            const std::array<float, 3> zenith{0.43f, 0.62f, 0.95f};

            std::array<float, 3> gradient{};
            if (y < 0.0f) {
                const float t = saturate((y + 0.12f) / 0.12f);
                for (int c = 0; c < 3; ++c) {
                    gradient[static_cast<size_t>(c)] =
                        below[static_cast<size_t>(c)] * (1.0f - t) + horizon[static_cast<size_t>(c)] * t;
                }
            } else {
                const float t = std::pow(saturate(y), 0.55f);
                for (int c = 0; c < 3; ++c) {
                    gradient[static_cast<size_t>(c)] =
                        horizon[static_cast<size_t>(c)] * (1.0f - t) + zenith[static_cast<size_t>(c)] * t;
                }
            }

            std::array<float, 3> result{};
            for (int c = 0; c < 3; ++c) {
                const float base = learned_initial_color(base_color[static_cast<size_t>(c)]);
                result[static_cast<size_t>(c)] = clamp_unit_color(0.35f * base + 0.65f * gradient[static_cast<size_t>(c)]);
            }
            return result;
        }

        void direction_to_equirect_uv(const float x, const float y, const float z, float& u, float& v) {
            const float longitude = std::atan2(x, z);
            const float latitude = std::asin(std::clamp(y, -1.0f, 1.0f));
            u = longitude / kTwoPi + 0.5f;
            v = 0.5f - latitude / kPi;
        }

        float sample_texture_logits_cpu(
            const float* texture,
            const int texture_height,
            const int texture_width,
            const int channel,
            float u,
            float v) {
            u = u - std::floor(u);
            v = saturate(v);
            const float tx = u * static_cast<float>(texture_width) - 0.5f;
            const float ty = v * static_cast<float>(texture_height - 1);
            const int x0 = static_cast<int>(std::floor(tx));
            const int y0 = static_cast<int>(std::floor(ty));
            const float wx = tx - static_cast<float>(x0);
            const float wy = ty - static_cast<float>(y0);
            const int x0w = (x0 % texture_width + texture_width) % texture_width;
            const int x1w = (x0w + 1) % texture_width;
            const int y0c = std::clamp(y0, 0, texture_height - 1);
            const int y1c = std::min(y0c + 1, texture_height - 1);
            const int plane = texture_height * texture_width;
            const int base = channel * plane;
            const float v00 = texture[base + y0c * texture_width + x0w];
            const float v10 = texture[base + y0c * texture_width + x1w];
            const float v01 = texture[base + y1c * texture_width + x0w];
            const float v11 = texture[base + y1c * texture_width + x1w];
            return (1.0f - wy) * ((1.0f - wx) * v00 + wx * v10) +
                   wy * ((1.0f - wx) * v01 + wx * v11);
        }

        std::array<float, 3> evaluate_lobes_cpu(
            const float* dirs,
            const float* colors,
            const int lobe_count,
            const float sharpness,
            const float x,
            const float y,
            const float z) {
            float max_dot = -1.0e20f;
            for (int i = 0; i < lobe_count; ++i) {
                const size_t base = static_cast<size_t>(i) * 3UL;
                const float dot = x * dirs[base + 0] + y * dirs[base + 1] + z * dirs[base + 2];
                max_dot = std::max(max_dot, dot);
            }

            std::array<float, 3> accum{0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;
            for (int i = 0; i < lobe_count; ++i) {
                const size_t base = static_cast<size_t>(i) * 3UL;
                const float dot = x * dirs[base + 0] + y * dirs[base + 1] + z * dirs[base + 2];
                const float weight = std::exp(sharpness * (dot - max_dot));
                weight_sum += weight;
                accum[0] += weight * colors[base + 0];
                accum[1] += weight * colors[base + 1];
                accum[2] += weight * colors[base + 2];
            }

            const float inv = weight_sum > 1.0e-8f ? 1.0f / weight_sum : 1.0f;
            return {accum[0] * inv, accum[1] * inv, accum[2] * inv};
        }

        lfs::core::Tensor make_lobe_dirs_tensor(const std::vector<float>& dirs) {
            auto dirs_cpu = lfs::core::Tensor::empty(
                {dirs.size() / 3UL, 3UL},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            std::copy(dirs.begin(), dirs.end(), dirs_cpu.ptr<float>());
            return dirs_cpu.cuda();
        }

        lfs::core::Tensor make_initial_lobe_logits(
            const std::vector<float>& dirs,
            const std::array<float, 3>& color,
            const std::array<float, 3>& up_axis) {
            const size_t lobe_count = dirs.size() / 3UL;
            const auto up = resolve_up_axis(up_axis);
            auto logits_cpu = lfs::core::Tensor::empty(
                {lobe_count, 3UL},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            auto* logits = logits_cpu.ptr<float>();
            for (size_t i = 0; i < lobe_count; ++i) {
                const float local_up =
                    up[0] * dirs[i * 3UL + 0] +
                    up[1] * dirs[i * 3UL + 1] +
                    up[2] * dirs[i * 3UL + 2];
                const auto sky_color = default_sky_color_for_dir(color, local_up);
                logits[i * 3UL + 0] = logit_unit_color(sky_color[0]);
                logits[i * 3UL + 1] = logit_unit_color(sky_color[1]);
                logits[i * 3UL + 2] = logit_unit_color(sky_color[2]);
            }
            return logits_cpu.cuda();
        }
    } // namespace

    void DirectionalBackground::initialize(const std::array<float, 3>& color, const int degree, const std::array<float, 3>& up_axis) {
        degree_ = clamp_degree(degree);
        lobe_count_ = lobe_count_for_degree(degree_);
        snapshot_texture_width_ = snapshot_width_for_degree(degree_);
        snapshot_texture_height_ = snapshot_height_for_degree(degree_);
        up_axis_ = resolve_up_axis(up_axis);

        const auto dirs = make_lobe_dirs(lobe_count_, up_axis_);
        lobe_dirs_ = make_lobe_dirs_tensor(dirs);
        lobe_logits_ = make_initial_lobe_logits(dirs, color, up_axis_);
        grad_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
        render_buffer_ = {};
        render_width_ = 0;
        render_height_ = 0;
        step_ = 0;

        LOG_INFO("Learned sky Gaussian lobes initialized: detail {}, {} lobes, preview {}x{}, up ({:.3f}, {:.3f}, {:.3f})",
                 degree_, lobe_count_, snapshot_texture_width_, snapshot_texture_height_,
                 up_axis_[0], up_axis_[1], up_axis_[2]);
    }

    void DirectionalBackground::ensure_initialized(const std::array<float, 3>& color, const int degree, const std::array<float, 3>& up_axis) {
        const int target_degree = clamp_degree(degree);
        const auto target_up = resolve_up_axis(up_axis);
        const float up_dot =
            target_up[0] * up_axis_[0] +
            target_up[1] * up_axis_[1] +
            target_up[2] * up_axis_[2];
        const bool up_changed = up_dot < 0.99999f;
        if (!is_initialized() ||
            degree_ != target_degree ||
            lobe_count_ != lobe_count_for_degree(target_degree) ||
            up_changed) {
            initialize(color, target_degree, target_up);
        } else if (step_ == 0) {
            const auto dirs = make_lobe_dirs(lobe_count_, up_axis_);
            lobe_logits_ = make_initial_lobe_logits(dirs, color, up_axis_);
            grad_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
            exp_avg_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
            exp_avg_sq_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
        }
    }

    std::optional<DirectionalBackgroundSnapshot> DirectionalBackground::snapshot() const {
        if (!is_initialized()) {
            return std::nullopt;
        }

        lfs::core::Tensor dirs_cpu = lobe_dirs_;
        lfs::core::Tensor logits_cpu = lobe_logits_;
        if (dirs_cpu.device() != lfs::core::Device::CPU) {
            dirs_cpu = dirs_cpu.cpu();
        }
        if (logits_cpu.device() != lfs::core::Device::CPU) {
            logits_cpu = logits_cpu.cpu();
        }
        dirs_cpu = dirs_cpu.is_contiguous() ? dirs_cpu : dirs_cpu.contiguous();
        logits_cpu = logits_cpu.is_contiguous() ? logits_cpu : logits_cpu.contiguous();

        DirectionalBackgroundSnapshot result;
        result.degree = degree_;
        result.texture_width = snapshot_texture_width_;
        result.texture_height = snapshot_texture_height_;
        result.lobe_count = lobe_count_;
        result.step = step_;

        const auto* dirs = dirs_cpu.ptr<float>();
        const auto* logits = logits_cpu.ptr<float>();
        result.lobe_dirs.assign(dirs, dirs + static_cast<size_t>(lobe_count_) * 3UL);
        result.lobe_colors.resize(static_cast<size_t>(lobe_count_) * 3UL);
        for (int i = 0; i < lobe_count_; ++i) {
            const size_t base = static_cast<size_t>(i) * 3UL;
            result.lobe_colors[base + 0] = sigmoid_clamped(logits[base + 0]);
            result.lobe_colors[base + 1] = sigmoid_clamped(logits[base + 1]);
            result.lobe_colors[base + 2] = sigmoid_clamped(logits[base + 2]);
        }

        const size_t pixel_count =
            static_cast<size_t>(snapshot_texture_width_) * static_cast<size_t>(snapshot_texture_height_);
        result.texture.resize(pixel_count * 3UL);
        const float sharpness = lobe_sharpness_for_degree(degree_);
        for (int y = 0; y < snapshot_texture_height_; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(snapshot_texture_height_);
            const float latitude = (0.5f - v) * kPi;
            const float cos_latitude = std::cos(latitude);
            for (int x = 0; x < snapshot_texture_width_; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(snapshot_texture_width_);
                const float longitude = (u - 0.5f) * kTwoPi;
                const auto color = evaluate_lobes_cpu(
                    dirs,
                    result.lobe_colors.data(),
                    lobe_count_,
                    sharpness,
                    std::sin(longitude) * cos_latitude,
                    std::sin(latitude),
                    -std::cos(longitude) * cos_latitude);
                const size_t dst =
                    (static_cast<size_t>(y) * static_cast<size_t>(snapshot_texture_width_) + static_cast<size_t>(x)) * 3UL;
                result.texture[dst + 0] = color[0];
                result.texture[dst + 1] = color[1];
                result.texture[dst + 2] = color[2];
            }
        }
        return result;
    }

    lfs::core::Tensor DirectionalBackground::render(
        const lfs::core::Camera& camera,
        const int width,
        const int height,
        const int tile_x_offset,
        const int tile_y_offset,
        const int full_width,
        const int full_height) {
        if (!is_initialized()) {
            throw std::runtime_error("DirectionalBackground::render called before initialization");
        }

        const int resolved_full_width = full_width > 0 ? full_width : width;
        const int resolved_full_height = full_height > 0 ? full_height : height;

        const lfs::core::TensorShape shape{
            3UL,
            static_cast<size_t>(height),
            static_cast<size_t>(width)};
        if (!render_buffer_.is_valid() || render_width_ != width || render_height_ != height || render_buffer_.shape() != shape) {
            render_buffer_ = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            render_width_ = width;
            render_height_ = height;
        }

        const auto [fx, fy, cx, cy] = camera.get_intrinsics();
        const cudaStream_t stream = lobe_logits_.stream();
        render_buffer_.set_stream(stream);
        lfs::training::kernels::launch_render_directional_background_lobes(
            lobe_dirs_.ptr<float>(),
            lobe_logits_.ptr<float>(),
            render_buffer_.ptr<float>(),
            lobe_count_,
            lobe_sharpness_for_degree(degree_),
            height,
            width,
            resolved_full_height,
            resolved_full_width,
            tile_x_offset,
            tile_y_offset,
            fx,
            fy,
            cx,
            cy,
            camera.world_view_transform_ptr(),
            static_cast<int>(camera.camera_model_type()),
            stream);

        return render_buffer_;
    }

    void DirectionalBackground::accumulate_gradient(
        const lfs::core::Camera& camera,
        const lfs::core::Tensor& grad_image,
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& sky_gate,
        const int tile_x_offset,
        const int tile_y_offset,
        const int full_width,
        const int full_height) {
        if (!is_initialized()) {
            return;
        }
        if (!grad_image.is_valid() || grad_image.ndim() != 3 || grad_image.shape()[0] != 3) {
            throw std::runtime_error("Directional background gradient expects CHW RGB grad_image");
        }
        if (!alpha.is_valid() || alpha.numel() == 0) {
            throw std::runtime_error("Directional background gradient requires alpha");
        }

        auto grad = contiguous_cuda(grad_image);
        auto alpha_cuda = contiguous_cuda(alpha);
        lfs::core::Tensor sky_gate_cuda;
        const float* sky_gate_ptr = nullptr;
        if (sky_gate.is_valid() && sky_gate.numel() > 0) {
            sky_gate_cuda = contiguous_cuda(sky_gate);
            sky_gate_ptr = sky_gate_cuda.ptr<float>();
        }

        const int height = static_cast<int>(grad.shape()[1]);
        const int width = static_cast<int>(grad.shape()[2]);
        const int resolved_full_width = full_width > 0 ? full_width : width;
        const int resolved_full_height = full_height > 0 ? full_height : height;
        const auto [fx, fy, cx, cy] = camera.get_intrinsics();
        const cudaStream_t stream = grad.stream();
        grad_.set_stream(stream);

        lfs::training::kernels::launch_accumulate_directional_background_lobe_grad(
            lobe_dirs_.ptr<float>(),
            lobe_logits_.ptr<float>(),
            grad.ptr<float>(),
            alpha_cuda.ptr<float>(),
            sky_gate_ptr,
            grad_.ptr<float>(),
            lobe_count_,
            lobe_sharpness_for_degree(degree_),
            height,
            width,
            resolved_full_height,
            resolved_full_width,
            tile_x_offset,
            tile_y_offset,
            fx,
            fy,
            cx,
            cy,
            camera.world_view_transform_ptr(),
            static_cast<int>(camera.camera_model_type()),
            stream);
    }

    void DirectionalBackground::add_l2_gradient(const float weight) {
        if (!is_initialized() || weight <= 0.0f) {
            return;
        }
        const cudaStream_t stream = lobe_logits_.stream();
        grad_.set_stream(stream);
        lfs::training::kernels::launch_directional_background_lobe_smoothness_grad(
            lobe_dirs_.ptr<float>(),
            lobe_logits_.ptr<float>(),
            grad_.ptr<float>(),
            lobe_count_,
            weight,
            stream);
    }

    void DirectionalBackground::optimizer_step(
        const float lr,
        const float beta1,
        const float beta2,
        const float eps) {
        if (!is_initialized()) {
            return;
        }
        if (lr <= 0.0f) {
            zero_grad();
            return;
        }

        ++step_;
        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, static_cast<double>(step_)));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, static_cast<double>(step_)));

        fast_lfs::optimizer::adam_step_raw(
            lobe_logits_.ptr<float>(),
            exp_avg_.ptr<float>(),
            exp_avg_sq_.ptr<float>(),
            grad_.ptr<float>(),
            static_cast<int>(lobe_logits_.numel()),
            lr,
            beta1,
            beta2,
            eps,
            static_cast<float>(bias_correction1_rcp),
            static_cast<float>(bias_correction2_sqrt_rcp));
        zero_grad();
    }

    void DirectionalBackground::zero_grad() {
        if (grad_.is_valid()) {
            grad_.zero_();
        }
    }

    void DirectionalBackground::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&DIRECTIONAL_BG_MAGIC), sizeof(DIRECTIONAL_BG_MAGIC));
        os.write(reinterpret_cast<const char*>(&DIRECTIONAL_BG_VERSION), sizeof(DIRECTIONAL_BG_VERSION));
        os.write(reinterpret_cast<const char*>(&degree_), sizeof(degree_));
        os.write(reinterpret_cast<const char*>(&lobe_count_), sizeof(lobe_count_));
        os.write(reinterpret_cast<const char*>(&snapshot_texture_width_), sizeof(snapshot_texture_width_));
        os.write(reinterpret_cast<const char*>(&snapshot_texture_height_), sizeof(snapshot_texture_height_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os << lobe_dirs_ << lobe_logits_ << exp_avg_ << exp_avg_sq_;
    }

    void DirectionalBackground::deserialize(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != DIRECTIONAL_BG_MAGIC) {
            throw std::runtime_error("Invalid DirectionalBackground checkpoint");
        }
        if (version == 0 || version > DIRECTIONAL_BG_VERSION) {
            throw std::runtime_error("Unsupported DirectionalBackground checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&degree_), sizeof(degree_));
        degree_ = clamp_degree(degree_);
        snapshot_texture_width_ = snapshot_width_for_degree(degree_);
        snapshot_texture_height_ = snapshot_height_for_degree(degree_);
        lobe_count_ = lobe_count_for_degree(degree_);

        if (version >= 4) {
            is.read(reinterpret_cast<char*>(&lobe_count_), sizeof(lobe_count_));
            is.read(reinterpret_cast<char*>(&snapshot_texture_width_), sizeof(snapshot_texture_width_));
            is.read(reinterpret_cast<char*>(&snapshot_texture_height_), sizeof(snapshot_texture_height_));
            is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
            is >> lobe_dirs_ >> lobe_logits_ >> exp_avg_ >> exp_avg_sq_;
        } else if (version == 3) {
            int texture_width = 0;
            int texture_height = 0;
            is.read(reinterpret_cast<char*>(&texture_width), sizeof(texture_width));
            is.read(reinterpret_cast<char*>(&texture_height), sizeof(texture_height));
            is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
            lfs::core::Tensor texture_logits;
            is >> texture_logits >> exp_avg_ >> exp_avg_sq_;

            lfs::core::Tensor texture_cpu = texture_logits.device() == lfs::core::Device::CPU ? texture_logits : texture_logits.cpu();
            texture_cpu = texture_cpu.is_contiguous() ? texture_cpu : texture_cpu.contiguous();
            const auto dirs = make_lobe_dirs(lobe_count_, {0.0f, 1.0f, 0.0f});
            auto logits_cpu = lfs::core::Tensor::empty(
                {static_cast<size_t>(lobe_count_), 3UL},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            auto* logits = logits_cpu.ptr<float>();
            const auto* texture = texture_cpu.ptr<float>();
            for (int i = 0; i < lobe_count_; ++i) {
                const size_t base = static_cast<size_t>(i) * 3UL;
                float u = 0.0f;
                float v = 0.0f;
                direction_to_equirect_uv(dirs[base + 0], dirs[base + 1], dirs[base + 2], u, v);
                for (int c = 0; c < 3; ++c) {
                    logits[base + static_cast<size_t>(c)] =
                        sample_texture_logits_cpu(texture, texture_height, texture_width, c, u, v);
                }
            }
            lobe_dirs_ = make_lobe_dirs_tensor(dirs);
            lobe_logits_ = logits_cpu;
            exp_avg_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CPU);
            exp_avg_sq_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CPU);
        } else {
            is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
            lfs::core::Tensor old_coeffs;
            is >> old_coeffs >> exp_avg_ >> exp_avg_sq_;
            lfs::core::Tensor coeffs_cpu = old_coeffs.device() == lfs::core::Device::CPU ? old_coeffs : old_coeffs.cpu();
            coeffs_cpu = coeffs_cpu.is_contiguous() ? coeffs_cpu : coeffs_cpu.contiguous();
            const auto* coeffs = coeffs_cpu.ptr<float>();
            const std::array<float, 3> logits{
                version == 1 ? logit_unit_color(coeffs[0]) : coeffs[0],
                version == 1 ? logit_unit_color(coeffs[1]) : coeffs[1],
                version == 1 ? logit_unit_color(coeffs[2]) : coeffs[2],
            };

            const auto dirs = make_lobe_dirs(lobe_count_, {0.0f, 1.0f, 0.0f});
            auto logits_cpu = lfs::core::Tensor::empty(
                {static_cast<size_t>(lobe_count_), 3UL},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            auto* lobe_logits = logits_cpu.ptr<float>();
            for (int i = 0; i < lobe_count_; ++i) {
                const size_t base = static_cast<size_t>(i) * 3UL;
                lobe_logits[base + 0] = logits[0];
                lobe_logits[base + 1] = logits[1];
                lobe_logits[base + 2] = logits[2];
            }
            lobe_dirs_ = make_lobe_dirs_tensor(dirs);
            lobe_logits_ = logits_cpu;
            exp_avg_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CPU);
            exp_avg_sq_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CPU);
        }

        lobe_dirs_ = lobe_dirs_.cuda();
        lobe_logits_ = lobe_logits_.cuda();
        exp_avg_ = exp_avg_.cuda();
        exp_avg_sq_ = exp_avg_sq_.cuda();
        grad_ = lfs::core::Tensor::zeros(lobe_logits_.shape(), lfs::core::Device::CUDA);
        render_buffer_ = {};
        render_width_ = 0;
        render_height_ = 0;

        LOG_INFO("Learned sky Gaussian lobes restored: detail {}, {} lobes, step {}", degree_, lobe_count_, step_);
    }

} // namespace lfs::training
