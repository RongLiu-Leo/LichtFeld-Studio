/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "directional_background.hpp"

#include "adam_api.h"
#include "core/logger.hpp"
#include "training/kernels/directional_background.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t DIRECTIONAL_BG_MAGIC = 0x4C464244; // "LFBD"
        constexpr uint32_t DIRECTIONAL_BG_VERSION = 2;

        int clamp_degree(const int degree) {
            return std::clamp(degree, 0, 2);
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

        lfs::core::Tensor contiguous_cuda(lfs::core::Tensor tensor) {
            if (tensor.device() != lfs::core::Device::CUDA) {
                tensor = tensor.cuda();
            }
            return tensor.is_contiguous() ? tensor : tensor.contiguous();
        }
    } // namespace

    void DirectionalBackground::initialize(const std::array<float, 3>& color, const int degree) {
        degree_ = clamp_degree(degree);
        const int bases = basis_count();

        auto coeffs_cpu = lfs::core::Tensor::zeros(
            {static_cast<size_t>(bases), 3UL},
            lfs::core::Device::CPU,
            lfs::core::DataType::Float32);
        auto* coeffs = coeffs_cpu.ptr<float>();
        coeffs[0] = logit_unit_color(learned_initial_color(color[0]));
        coeffs[1] = logit_unit_color(learned_initial_color(color[1]));
        coeffs[2] = logit_unit_color(learned_initial_color(color[2]));

        coeffs_ = coeffs_cpu.cuda();
        grad_ = lfs::core::Tensor::zeros(coeffs_.shape(), lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::zeros(coeffs_.shape(), lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros(coeffs_.shape(), lfs::core::Device::CUDA);
        render_buffer_ = {};
        render_width_ = 0;
        render_height_ = 0;
        step_ = 0;

        LOG_INFO("Learned directional background initialized: SH degree {}, {} coefficients, RGB({:.2f}, {:.2f}, {:.2f})",
                 degree_, bases, color[0], color[1], color[2]);
    }

    void DirectionalBackground::ensure_initialized(const std::array<float, 3>& color, const int degree) {
        const int target_degree = clamp_degree(degree);
        if (!is_initialized() || degree_ != target_degree) {
            initialize(color, target_degree);
        } else if (step_ == 0) {
            const std::array<float, 3> logits{
                logit_unit_color(learned_initial_color(color[0])),
                logit_unit_color(learned_initial_color(color[1])),
                logit_unit_color(learned_initial_color(color[2])),
            };
            cudaMemcpyAsync(
                coeffs_.ptr<float>(),
                logits.data(),
                sizeof(float) * logits.size(),
                cudaMemcpyHostToDevice,
                coeffs_.stream());
        }
    }

    std::optional<DirectionalBackgroundSnapshot> DirectionalBackground::snapshot() const {
        if (!is_initialized()) {
            return std::nullopt;
        }

        lfs::core::Tensor coeffs_cpu = coeffs_;
        if (coeffs_cpu.device() != lfs::core::Device::CPU) {
            coeffs_cpu = coeffs_cpu.cpu();
        }
        coeffs_cpu = coeffs_cpu.is_contiguous() ? coeffs_cpu : coeffs_cpu.contiguous();

        DirectionalBackgroundSnapshot result;
        result.degree = degree_;
        result.step = step_;
        const auto* coeffs = coeffs_cpu.ptr<float>();
        const int bases = std::min(basis_count(), static_cast<int>(result.coeffs.size()));
        for (int b = 0; b < bases; ++b) {
            result.coeffs[static_cast<size_t>(b)] = {
                coeffs[b * 3 + 0],
                coeffs[b * 3 + 1],
                coeffs[b * 3 + 2],
            };
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
        const cudaStream_t stream = coeffs_.stream();
        render_buffer_.set_stream(stream);
        lfs::training::kernels::launch_render_directional_background_sh(
            coeffs_.ptr<float>(),
            render_buffer_.ptr<float>(),
            degree_,
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

        lfs::training::kernels::launch_accumulate_directional_background_sh_grad(
            coeffs_.ptr<float>(),
            grad.ptr<float>(),
            alpha_cuda.ptr<float>(),
            sky_gate_ptr,
            grad_.ptr<float>(),
            degree_,
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
        const cudaStream_t stream = coeffs_.stream();
        grad_.set_stream(stream);
        lfs::training::kernels::launch_directional_background_l2_grad(
            coeffs_.ptr<float>(), grad_.ptr<float>(), degree_, weight, stream);
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
            coeffs_.ptr<float>(),
            exp_avg_.ptr<float>(),
            exp_avg_sq_.ptr<float>(),
            grad_.ptr<float>(),
            static_cast<int>(coeffs_.numel()),
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
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os << coeffs_ << exp_avg_ << exp_avg_sq_;
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
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is >> coeffs_ >> exp_avg_ >> exp_avg_sq_;
        if (version == 1 && coeffs_.is_valid()) {
            lfs::core::Tensor coeffs_cpu = coeffs_.device() == lfs::core::Device::CPU ? coeffs_ : coeffs_.cpu();
            coeffs_cpu = coeffs_cpu.is_contiguous() ? coeffs_cpu : coeffs_cpu.contiguous();
            auto* coeffs = coeffs_cpu.ptr<float>();
            for (size_t i = 0; i < coeffs_cpu.numel(); ++i) {
                coeffs[i] = logit_unit_color(coeffs[i]);
            }
            coeffs_ = coeffs_cpu;
            exp_avg_ = lfs::core::Tensor::zeros(coeffs_.shape(), lfs::core::Device::CPU);
            exp_avg_sq_ = lfs::core::Tensor::zeros(coeffs_.shape(), lfs::core::Device::CPU);
        }
        coeffs_ = coeffs_.cuda();
        exp_avg_ = exp_avg_.cuda();
        exp_avg_sq_ = exp_avg_sq_.cuda();
        grad_ = lfs::core::Tensor::zeros(coeffs_.shape(), lfs::core::Device::CUDA);
        render_buffer_ = {};
        render_width_ = 0;
        render_height_ = 0;

        LOG_INFO("Learned directional background restored: SH degree {}, step {}", degree_, step_);
    }

} // namespace lfs::training
