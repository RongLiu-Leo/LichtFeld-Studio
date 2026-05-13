/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sky_discriminator.hpp"

#include "adam_api.h"
#include "core/logger.hpp"
#include "training/kernels/sky_discriminator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr int F = lfs::training::kernels::kSkyDiscFeatureDim;
        constexpr int Hdim = lfs::training::kernels::kSkyDiscHiddenSize;
        constexpr int N_PARAMS = lfs::training::kernels::kSkyDiscParamCount;
        constexpr int kW1Offset = 0;
        constexpr int kB1Offset = kW1Offset + F * Hdim;
        constexpr int kW2Offset = kB1Offset + Hdim;
        constexpr int kB2Offset = kW2Offset + Hdim;

        lfs::core::Tensor contiguous_cuda(lfs::core::Tensor tensor) {
            if (tensor.device() != lfs::core::Device::CUDA) {
                tensor = tensor.cuda();
            }
            return tensor.is_contiguous() ? tensor : tensor.contiguous();
        }

        std::vector<float> sample_initial_weights() {
            std::vector<float> w(static_cast<size_t>(N_PARAMS), 0.0f);
            std::mt19937 rng(0xC0FFEEu);
            // He-uniform for layer 1 (ReLU); Xavier-uniform for layer 2 (sigmoid).
            const float lim1 = std::sqrt(6.0f / static_cast<float>(F));
            const float lim2 = std::sqrt(6.0f / static_cast<float>(Hdim + 1));
            std::uniform_real_distribution<float> d1(-lim1, lim1);
            std::uniform_real_distribution<float> d2(-lim2, lim2);
            for (int i = 0; i < F * Hdim; ++i) {
                w[static_cast<size_t>(kW1Offset + i)] = d1(rng);
            }
            for (int j = 0; j < Hdim; ++j) {
                w[static_cast<size_t>(kB1Offset + j)] = 0.0f;
            }
            for (int j = 0; j < Hdim; ++j) {
                w[static_cast<size_t>(kW2Offset + j)] = d2(rng);
            }
            // Bias the output toward "not sky" so the bootstrap output is
            // conservative if it ever leaks through before training.
            w[static_cast<size_t>(kB2Offset)] = -1.0f;
            return w;
        }
    } // namespace

    int SkyDiscriminator::scaled(const int v) const noexcept {
        const float s = steps_scaler_ > 0.0f ? steps_scaler_ : 1.0f;
        return std::max(1, static_cast<int>(std::lround(static_cast<float>(v) * s)));
    }

    void SkyDiscriminator::initialize(const float steps_scaler) {
        steps_scaler_ = steps_scaler > 0.0f ? steps_scaler : 1.0f;
        const auto init = sample_initial_weights();
        auto weights_cpu = lfs::core::Tensor::empty(
            {static_cast<size_t>(N_PARAMS)},
            lfs::core::Device::CPU,
            lfs::core::DataType::Float32);
        std::copy(init.begin(), init.end(), weights_cpu.ptr<float>());
        weights_ = weights_cpu.cuda();
        grad_ = lfs::core::Tensor::zeros(weights_.shape(), lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::zeros(weights_.shape(), lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros(weights_.shape(), lfs::core::Device::CUDA);
        label_buffer_ = {};
        sample_weight_buffer_ = {};
        label_buffer_height_ = 0;
        label_buffer_width_ = 0;
        step_ = 0;
        frozen_logged_ = false;
        LOG_INFO("SkyDiscriminator initialized: {} params ({} hidden units, {} features), schedule scaled by {:.3f} (bootstrap={}, warmup={}, freeze={})",
                 N_PARAMS, Hdim, F, steps_scaler_,
                 scaled(kBootstrapIters), scaled(kWarmupIters), scaled(kTrainCutoffIters));
    }

    bool SkyDiscriminator::is_initialized() const noexcept {
        return weights_.is_valid() && weights_.numel() == static_cast<size_t>(N_PARAMS);
    }

    bool SkyDiscriminator::is_active(const int iter) const noexcept {
        return is_initialized() && iter >= scaled(kBootstrapIters);
    }

    bool SkyDiscriminator::is_training_phase(const int iter) const noexcept {
        return is_active(iter) && iter < scaled(kTrainCutoffIters);
    }

    float SkyDiscriminator::chroma_blend(const int iter) const noexcept {
        const int boot = scaled(kBootstrapIters);
        const int warm_total = scaled(kWarmupIters);
        if (iter < boot) {
            return 1.0f;
        }
        const int warm = iter - boot;
        if (warm >= warm_total) {
            return 0.0f;
        }
        const float t = static_cast<float>(warm) / static_cast<float>(warm_total);
        return std::clamp(1.0f - t, 0.0f, 1.0f);
    }

    void SkyDiscriminator::forward(
        const lfs::core::Tensor& target_image,
        const bool target_is_chw,
        const lfs::core::Tensor& rendered_alpha,
        const lfs::core::Tensor& chroma_prior,
        const int tile_y_offset,
        const int full_height,
        lfs::core::Tensor& sky_gate,
        const int iter) {
        if (!is_active(iter) ||
            !sky_gate.is_valid() ||
            sky_gate.numel() == 0) {
            return;
        }
        if (!target_image.is_valid() || target_image.numel() == 0) {
            return;
        }
        if (!rendered_alpha.is_valid() || rendered_alpha.numel() == 0) {
            return;
        }

        auto tgt = contiguous_cuda(target_image);
        auto alpha = contiguous_cuda(rendered_alpha);
        const float* chroma_ptr = nullptr;
        lfs::core::Tensor chroma_cuda;
        if (chroma_prior.is_valid() && chroma_prior.numel() == sky_gate.numel()) {
            chroma_cuda = contiguous_cuda(chroma_prior);
            chroma_ptr = chroma_cuda.ptr<float>();
        }

        int height = 0;
        int width = 0;
        if (sky_gate.ndim() == 2) {
            height = static_cast<int>(sky_gate.shape()[0]);
            width = static_cast<int>(sky_gate.shape()[1]);
        } else {
            return;
        }

        const cudaStream_t stream = sky_gate.stream();
        lfs::training::kernels::launch_sky_discriminator_forward(
            tgt.ptr<float>(),
            target_is_chw,
            alpha.ptr<float>(),
            chroma_ptr,
            weights_.ptr<float>(),
            height,
            width,
            full_height > 0 ? full_height : height,
            tile_y_offset,
            chroma_blend(iter),
            sky_gate.ptr<float>(),
            stream);
    }

    bool SkyDiscriminator::train_step(
        const lfs::core::Tensor& target_image,
        const bool target_is_chw,
        const lfs::core::Tensor& rendered_alpha,
        const lfs::core::Tensor& chroma_prior,
        const int tile_y_offset,
        const int full_height,
        const int iter) {
        if (!is_training_phase(iter)) {
            if (is_active(iter) && step_ > 0 && !frozen_logged_) {
                LOG_INFO("SkyDiscriminator frozen at iter {}: {} train steps; forward-only from now on",
                         iter, step_);
                frozen_logged_ = true;
            }
            return false;
        }
        if (!target_image.is_valid() ||
            !rendered_alpha.is_valid() ||
            !chroma_prior.is_valid()) {
            return false;
        }

        auto tgt = contiguous_cuda(target_image);
        auto alpha = contiguous_cuda(rendered_alpha);
        auto chroma = contiguous_cuda(chroma_prior);

        int height = 0;
        int width = 0;
        if (alpha.ndim() == 2) {
            height = static_cast<int>(alpha.shape()[0]);
            width = static_cast<int>(alpha.shape()[1]);
        } else if (alpha.ndim() == 3 && alpha.shape()[0] == 1) {
            height = static_cast<int>(alpha.shape()[1]);
            width = static_cast<int>(alpha.shape()[2]);
        } else {
            return false;
        }

        const cudaStream_t stream = weights_.stream();

        if (!label_buffer_.is_valid() ||
            label_buffer_height_ != height ||
            label_buffer_width_ != width) {
            label_buffer_ = lfs::core::Tensor::empty(
                {static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            sample_weight_buffer_ = lfs::core::Tensor::empty(
                {static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            label_buffer_height_ = height;
            label_buffer_width_ = width;
        }
        label_buffer_.set_stream(stream);
        sample_weight_buffer_.set_stream(stream);

        lfs::training::kernels::launch_sky_discriminator_pseudo_labels(
            tgt.ptr<float>(),
            target_is_chw,
            alpha.ptr<float>(),
            chroma.ptr<float>(),
            height,
            width,
            full_height > 0 ? full_height : height,
            tile_y_offset,
            label_buffer_.ptr<float>(),
            sample_weight_buffer_.ptr<float>(),
            stream);

        grad_.set_stream(stream);
        grad_.zero_();

        // 1 / (H*W) keeps the gradient magnitude comparable across resolutions.
        const float loss_scale = 1.0f / std::max(1.0f, static_cast<float>(height * width));
        lfs::training::kernels::launch_sky_discriminator_backward(
            tgt.ptr<float>(),
            target_is_chw,
            alpha.ptr<float>(),
            chroma.ptr<float>(),
            weights_.ptr<float>(),
            label_buffer_.ptr<float>(),
            sample_weight_buffer_.ptr<float>(),
            height,
            width,
            full_height > 0 ? full_height : height,
            tile_y_offset,
            loss_scale,
            grad_.ptr<float>(),
            stream);

        ++step_;
        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(static_cast<double>(kBeta1), static_cast<double>(step_)));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(static_cast<double>(kBeta2), static_cast<double>(step_)));
        fast_lfs::optimizer::adam_step_raw(
            weights_.ptr<float>(),
            exp_avg_.ptr<float>(),
            exp_avg_sq_.ptr<float>(),
            grad_.ptr<float>(),
            static_cast<int>(weights_.numel()),
            kLearningRate,
            kBeta1,
            kBeta2,
            kEps,
            static_cast<float>(bias_correction1_rcp),
            static_cast<float>(bias_correction2_sqrt_rcp));
        return true;
    }

    void SkyDiscriminator::zero_grad() {
        if (grad_.is_valid()) {
            grad_.zero_();
        }
    }

} // namespace lfs::training
