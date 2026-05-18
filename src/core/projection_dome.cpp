/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/projection_dome.hpp"

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    namespace {

        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kHalfPi = kPi * 0.5f;
        constexpr float kTwoPi = kPi * 2.0f;
        // The training/viewer data basis follows COLMAP camera convention where +Y is down.
        // Visual sky-up is therefore the negative-Y half-space.
        constexpr float kSkyUpYSign = -1.0f;

        struct CubemapFaceDef {
            std::string_view id;
            std::string_view label;
        };

        constexpr std::array<CubemapFaceDef, 6> kCubemapFaces{{
            {"pos_x", "+X"},
            {"neg_x", "-X"},
            {"neg_y", "-Y"},
            {"pos_y", "+Y"},
            {"pos_z", "+Z"},
            {"neg_z", "-Z"},
        }};

        [[nodiscard]] int clampSegmentCount(const int value, const int minimum) {
            return std::max(value, minimum);
        }

        [[nodiscard]] uint8_t toByte(const float value) {
            return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
        }

        [[nodiscard]] std::vector<uint8_t> makePlaceholderTexture(int width, int height) {
            width = std::max(width, 2);
            height = std::max(height, 2);

            std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
            const int grid_x = std::max(8, width / 16);
            const int grid_y = std::max(8, height / 8);

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const float u = static_cast<float>(x) / static_cast<float>(std::max(1, width - 1));
                    const float v = static_cast<float>(y) / static_cast<float>(std::max(1, height - 1));
                    const bool grid = (x % grid_x) < 2 || (y % grid_y) < 2;
                    const glm::vec3 base = glm::mix(
                        glm::vec3(0.10f, 0.14f, 0.18f),
                        glm::vec3(0.34f, 0.42f, 0.50f),
                        0.5f * u + 0.5f * (1.0f - v));
                    const glm::vec3 color = grid ? glm::vec3(0.90f, 0.95f, 1.0f) : base;
                    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
                    pixels[idx + 0] = toByte(color.r);
                    pixels[idx + 1] = toByte(color.g);
                    pixels[idx + 2] = toByte(color.b);
                }
            }
            return pixels;
        }

        [[nodiscard]] std::optional<glm::vec3> tensorVec3(const Tensor& tensor) {
            if (!tensor.is_valid() || tensor.numel() < 3) {
                return std::nullopt;
            }

            const auto cpu = tensor.to(DataType::Float32).cpu().contiguous();
            const float* values = cpu.ptr<float>();
            const glm::vec3 result(values[0], values[1], values[2]);
            if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
                return std::nullopt;
            }
            return result;
        }

        [[nodiscard]] NodeId findDatasetParent(const Scene& scene) {
            for (const auto* node : scene.getNodes()) {
                if (node && node->type == NodeType::DATASET) {
                    return node->id;
                }
            }
            return NULL_NODE;
        }

        struct CameraPose {
            glm::mat3 R{1.0f};
            glm::vec3 T{0.0f};
            glm::vec3 local_center{0.0f};
        };

        [[nodiscard]] std::optional<CameraPose> readCameraPose(const Camera& camera) {
            if (!camera.R().is_valid() || !camera.T().is_valid() ||
                camera.R().numel() < 9 || camera.T().numel() < 3) {
                return std::nullopt;
            }

            auto R_cpu = camera.R().to(DataType::Float32).cpu().contiguous();
            auto T_cpu = camera.T().to(DataType::Float32).cpu().contiguous();
            auto R_acc = R_cpu.accessor<float, 2>();
            const float* T_ptr = T_cpu.ptr<float>();

            CameraPose pose;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    pose.R[col][row] = R_acc(row, col);
                }
                pose.T[static_cast<int>(row)] = T_ptr[row];
            }
            pose.local_center = -(glm::transpose(pose.R) * pose.T);
            return pose;
        }

        [[nodiscard]] glm::vec3 cameraWorldPosition(const Scene& scene, const SceneNode& node, const CameraPose& pose) {
            return glm::vec3(scene.getWorldTransform(node.id) * glm::vec4(pose.local_center, 1.0f));
        }

        [[nodiscard]] glm::vec3 toParentLocal(const Scene& scene, const NodeId parent_id, const glm::vec3& world_point) {
            if (parent_id == NULL_NODE) {
                return world_point;
            }
            const glm::mat4 parent_inv = glm::inverse(scene.getWorldTransform(parent_id));
            return glm::vec3(parent_inv * glm::vec4(world_point, 1.0f));
        }

        [[nodiscard]] glm::vec3 pointOnUnitDome(const float u, const float v) {
            const float phi = u * kTwoPi;
            const float theta = std::clamp(v, 0.0f, 1.0f) * kHalfPi;
            const float xy_radius = std::cos(theta);
            return glm::vec3(std::cos(phi) * xy_radius,
                             kSkyUpYSign * std::sin(theta),
                             std::sin(phi) * xy_radius);
        }

        [[nodiscard]] bool isSkyDirection(const glm::vec3& dir) {
            return kSkyUpYSign * dir.y >= 0.0f;
        }

        [[nodiscard]] float skyElevation01(const glm::vec3& dir) {
            return std::asin(std::clamp(kSkyUpYSign * dir.y, 0.0f, 1.0f)) / kHalfPi;
        }

        [[nodiscard]] glm::vec3 cubemapDirection(const std::string_view face_id,
                                                 const float a,
                                                 const float b) {
            if (face_id == "pos_x") {
                return glm::normalize(glm::vec3(1.0f, kSkyUpYSign * b, -a));
            }
            if (face_id == "neg_x") {
                return glm::normalize(glm::vec3(-1.0f, kSkyUpYSign * b, a));
            }
            if (face_id == "pos_y") {
                return glm::normalize(glm::vec3(a, -kSkyUpYSign, -b));
            }
            if (face_id == "neg_y") {
                return glm::normalize(glm::vec3(a, kSkyUpYSign, b));
            }
            if (face_id == "pos_z") {
                return glm::normalize(glm::vec3(a, kSkyUpYSign * b, 1.0f));
            }
            return glm::normalize(glm::vec3(-a, kSkyUpYSign * b, -1.0f));
        }

        [[nodiscard]] float edgeWeight(const float x,
                                       const float y,
                                       const int width,
                                       const int height,
                                       const float falloff,
                                       const bool wrap_x) {
            if (falloff <= 0.0f) {
                return 1.0f;
            }
            const float edge_x = wrap_x ? static_cast<float>(width)
                                        : std::min(x, static_cast<float>(width - 1) - x);
            const float edge_y = std::min(y, static_cast<float>(height - 1) - y);
            const float edge = std::min(edge_x, edge_y);
            const float feather = falloff * static_cast<float>(std::min(width, height));
            return std::clamp(edge / std::max(feather, 1.0f), 0.0f, 1.0f);
        }

        [[nodiscard]] glm::vec3 bilinearSample(const std::vector<uint8_t>& pixels,
                                               const int width,
                                               const int height,
                                               float x,
                                               float y,
                                               const bool wrap_x) {
            if (wrap_x) {
                x = std::fmod(x, static_cast<float>(width));
                if (x < 0.0f) {
                    x += static_cast<float>(width);
                }
            } else {
                x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
            }
            y = std::clamp(y, 0.0f, static_cast<float>(height - 1));

            const int x0 = static_cast<int>(std::floor(x));
            const int y0 = static_cast<int>(std::floor(y));
            const int x1 = wrap_x ? ((x0 + 1) % width) : std::min(x0 + 1, width - 1);
            const int y1 = std::min(y0 + 1, height - 1);
            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);

            const auto sample = [&](const int sx, const int sy) {
                const size_t idx = (static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)) * 3u;
                return glm::vec3(
                    static_cast<float>(pixels[idx + 0]) / 255.0f,
                    static_cast<float>(pixels[idx + 1]) / 255.0f,
                    static_cast<float>(pixels[idx + 2]) / 255.0f);
            };

            const glm::vec3 c00 = sample(x0, y0);
            const glm::vec3 c10 = sample(x1, y0);
            const glm::vec3 c01 = sample(x0, y1);
            const glm::vec3 c11 = sample(x1, y1);
            return glm::mix(glm::mix(c00, c10, tx), glm::mix(c01, c11, tx), ty);
        }

        [[nodiscard]] std::filesystem::path previewPath(const std::filesystem::path& output_dir,
                                                        const std::string_view face_id) {
            return output_dir / ("sky_preview_" + std::string(face_id) + ".png");
        }

        [[nodiscard]] std::filesystem::path maskPath(const std::filesystem::path& output_dir,
                                                     const std::string_view face_id) {
            return output_dir / ("sky_mask_" + std::string(face_id) + ".png");
        }

        [[nodiscard]] std::filesystem::path overlayPath(const std::filesystem::path& output_dir,
                                                        const std::string_view face_id) {
            return output_dir / ("sky_overlay_" + std::string(face_id) + ".png");
        }

        [[nodiscard]] bool saveRawImage(const std::filesystem::path& path,
                                        std::vector<uint8_t>& pixels,
                                        const int width,
                                        const int height,
                                        const int channels) {
            if (pixels.empty() || width <= 0 || height <= 0 || channels <= 0) {
                return false;
            }
            return save_img_data(path, std::make_tuple(pixels.data(), width, height, channels));
        }

        [[nodiscard]] glm::vec3 sampleDomeTexture(const TextureImage& texture, const glm::vec3& dir) {
            const float phi = std::atan2(dir.z, dir.x);
            float u = phi / kTwoPi;
            if (u < 0.0f) {
                u += 1.0f;
            }
            const float dome_v = skyElevation01(dir);
            const float x = u * static_cast<float>(texture.width);
            const float y = (1.0f - dome_v) * static_cast<float>(std::max(1, texture.height - 1));

            if (texture.channels >= 3) {
                return bilinearSample(texture.pixels, texture.width, texture.height, x, y, true);
            }

            const auto sample_channel = [&](float sx, float sy) {
                sx = std::fmod(sx, static_cast<float>(texture.width));
                if (sx < 0.0f) {
                    sx += static_cast<float>(texture.width);
                }
                sy = std::clamp(sy, 0.0f, static_cast<float>(texture.height - 1));
                const int ix = static_cast<int>(std::round(sx)) % texture.width;
                const int iy = std::clamp(static_cast<int>(std::round(sy)), 0, texture.height - 1);
                const size_t idx = (static_cast<size_t>(iy) * static_cast<size_t>(texture.width) + static_cast<size_t>(ix)) *
                                   static_cast<size_t>(std::max(1, texture.channels));
                const float v = texture.pixels[idx] / 255.0f;
                return glm::vec3(v);
            };
            return sample_channel(x, y);
        }

        [[nodiscard]] int countMarkedPixels(const std::vector<uint8_t>& mask) {
            return static_cast<int>(std::count_if(mask.begin(), mask.end(), [](const uint8_t v) {
                return v >= 128;
            }));
        }

        [[nodiscard]] std::vector<uint8_t> loadMaskPixels(const std::filesystem::path& mask_path,
                                                          const int face_size) {
            const size_t pixel_count = static_cast<size_t>(face_size) * static_cast<size_t>(face_size);
            std::vector<uint8_t> mask(pixel_count, 0);
            if (!std::filesystem::exists(mask_path)) {
                return mask;
            }

            unsigned char* data = nullptr;
            int width = 0;
            int height = 0;
            int channels = 0;
            try {
                std::tie(data, width, height, channels) = load_image(mask_path, -1, 0);
            } catch (const std::exception& e) {
                LOG_WARN("Sky mask load failed '{}': {}", path_to_utf8(mask_path), e.what());
                return mask;
            }

            if (!data || width != face_size || height != face_size || channels < 1) {
                if (data) {
                    free_image(data);
                }
                return mask;
            }

            for (size_t idx = 0; idx < pixel_count; ++idx) {
                mask[idx] = data[idx * static_cast<size_t>(channels)] >= 128 ? 255 : 0;
            }
            free_image(data);
            return mask;
        }

        [[nodiscard]] bool writeMaskAndOverlay(const std::filesystem::path& mask_path,
                                               const std::filesystem::path& overlay_path,
                                               const std::vector<uint8_t>& mask,
                                               const int face_size) {
            std::vector<uint8_t> mask_out = mask;
            if (!saveRawImage(mask_path, mask_out, face_size, face_size, 1)) {
                return false;
            }

            std::vector<uint8_t> overlay(mask.size() * 4u, 0);
            for (size_t idx = 0; idx < mask.size(); ++idx) {
                if (mask[idx] < 128) {
                    continue;
                }
                overlay[idx * 4u + 0] = 45;
                overlay[idx * 4u + 1] = 194;
                overlay[idx * 4u + 2] = 255;
                overlay[idx * 4u + 3] = 150;
            }
            return saveRawImage(overlay_path, overlay, face_size, face_size, 4);
        }

        [[nodiscard]] bool projectToImage(const Camera& camera,
                                          const CameraPose& pose,
                                          const glm::vec3& camera_data_point,
                                          const int image_width,
                                          const int image_height,
                                          float& out_x,
                                          float& out_y,
                                          bool& out_wrap_x) {
            out_wrap_x = false;
            const glm::vec3 camera_point = pose.R * camera_data_point + pose.T;

            if (camera.camera_model_type() == CameraModelType::EQUIRECTANGULAR) {
                const float len = glm::length(camera_point);
                if (len <= 1e-6f) {
                    return false;
                }
                const float azimuth = std::atan2(camera_point.x, camera_point.z);
                const float elevation = std::asin(std::clamp(camera_point.y / len, -1.0f, 1.0f));
                out_x = (azimuth / kTwoPi + 0.5f) * static_cast<float>(image_width);
                out_y = (elevation / kPi + 0.5f) * static_cast<float>(image_height);
                out_wrap_x = true;
                return out_y >= 0.0f && out_y <= static_cast<float>(image_height - 1);
            }

            if (camera.camera_model_type() != CameraModelType::PINHOLE) {
                return false;
            }

            if (camera_point.z <= 1e-5f) {
                return false;
            }

            const int camera_width = camera.camera_width() > 0 ? camera.camera_width() : image_width;
            const int camera_height = camera.camera_height() > 0 ? camera.camera_height() : image_height;
            const float sx = static_cast<float>(image_width) / static_cast<float>(std::max(1, camera_width));
            const float sy = static_cast<float>(image_height) / static_cast<float>(std::max(1, camera_height));
            out_x = camera.focal_x() * sx * (camera_point.x / camera_point.z) + camera.center_x() * sx;
            out_y = camera.focal_y() * sy * (camera_point.y / camera_point.z) + camera.center_y() * sy;
            return out_x >= 0.0f && out_x <= static_cast<float>(image_width - 1) &&
                   out_y >= 0.0f && out_y <= static_cast<float>(image_height - 1);
        }

        struct LoadedImage {
            std::vector<uint8_t> pixels;
            int width = 0;
            int height = 0;
            int channels = 3;
        };

        struct ProjectionSample {
            glm::vec3 color{0.0f};
            float score = 0.0f;
        };

        struct LoadedMask {
            std::vector<uint8_t> pixels;
            int width = 0;
            int height = 0;
            int masked_pixels = 0;
        };

        [[nodiscard]] std::optional<LoadedImage> loadCameraImage(const Camera& camera,
                                                                 const ProjectionDomeBakeOptions& options) {
            const auto& image_path = camera.image_path();
            if (image_path.empty() || !std::filesystem::exists(image_path)) {
                return std::nullopt;
            }

            unsigned char* data = nullptr;
            int width = 0;
            int height = 0;
            int channels = 0;
            try {
                std::tie(data, width, height, channels) =
                    load_image(image_path, options.resize_factor, options.max_image_width);
            } catch (const std::exception& e) {
                LOG_WARN("Projection dome skipped camera image '{}': {}",
                         path_to_utf8(image_path), e.what());
                return std::nullopt;
            }

            if (!data || width <= 1 || height <= 1 || channels < 3) {
                if (data) {
                    free_image(data);
                }
                return std::nullopt;
            }

            LoadedImage image;
            image.width = width;
            image.height = height;
            image.channels = 3;
            image.pixels.assign(data, data + static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
            free_image(data);
            return image;
        }

        [[nodiscard]] std::optional<LoadedImage> loadRawImage(const std::filesystem::path& path) {
            if (path.empty() || !std::filesystem::exists(path)) {
                return std::nullopt;
            }

            unsigned char* data = nullptr;
            int width = 0;
            int height = 0;
            int channels = 0;
            try {
                std::tie(data, width, height, channels) = load_image(path, -1, 0);
            } catch (const std::exception& e) {
                LOG_WARN("Image load failed '{}': {}", path_to_utf8(path), e.what());
                return std::nullopt;
            }

            if (!data || width <= 0 || height <= 0 || channels <= 0) {
                if (data) {
                    free_image(data);
                }
                return std::nullopt;
            }

            LoadedImage image;
            image.width = width;
            image.height = height;
            image.channels = channels;
            image.pixels.assign(data,
                                data + static_cast<size_t>(width) *
                                           static_cast<size_t>(height) *
                                           static_cast<size_t>(channels));
            free_image(data);
            return image;
        }

        [[nodiscard]] std::filesystem::path resolveManifestPath(const std::filesystem::path& manifest_path,
                                                                const nlohmann::json& value) {
            if (!value.is_string()) {
                return {};
            }
            std::filesystem::path path = utf8_to_path(value.get<std::string>());
            if (path.is_relative()) {
                path = manifest_path.parent_path() / path;
            }
            return path;
        }

        [[nodiscard]] glm::vec3 samplePreviewColor(const std::optional<LoadedImage>& preview,
                                                   const TextureImage& dome_texture,
                                                   const std::string_view face_id,
                                                   const int face_size,
                                                   const int x,
                                                   const int y,
                                                   const glm::vec3& dir) {
            if (preview && preview->width == face_size && preview->height == face_size &&
                preview->channels >= 3 && !preview->pixels.empty()) {
                const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(face_size) +
                                    static_cast<size_t>(x)) *
                                   static_cast<size_t>(preview->channels);
                return glm::vec3(
                    static_cast<float>(preview->pixels[idx + 0]) / 255.0f,
                    static_cast<float>(preview->pixels[idx + 1]) / 255.0f,
                    static_cast<float>(preview->pixels[idx + 2]) / 255.0f);
            }

            (void)face_id;
            if (dome_texture.width > 1 && dome_texture.height > 1 && !dome_texture.pixels.empty()) {
                return sampleDomeTexture(dome_texture, dir);
            }
            return glm::vec3(0.72f, 0.82f, 0.95f);
        }

        [[nodiscard]] std::optional<LoadedMask> loadCameraMask(const Camera& camera,
                                                               const ProjectionDomeBakeOptions& options) {
            if (!options.reject_masked_pixels || !camera.has_mask()) {
                return std::nullopt;
            }

            unsigned char* data = nullptr;
            int width = 0;
            int height = 0;
            int channels = 0;
            try {
                std::tie(data, width, height, channels) =
                    load_image(camera.mask_path(), options.resize_factor, options.max_image_width);
            } catch (const std::exception& e) {
                LOG_WARN("Projection dome skipped camera mask '{}': {}",
                         path_to_utf8(camera.mask_path()), e.what());
                return std::nullopt;
            }

            if (!data || width <= 1 || height <= 1 || channels < 1) {
                if (data) {
                    free_image(data);
                }
                return std::nullopt;
            }

            LoadedMask mask;
            mask.width = width;
            mask.height = height;
            mask.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
            const float threshold = (options.mask_threshold > 0.0f && options.mask_threshold < 1.0f)
                                        ? options.mask_threshold
                                        : 0.5f;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                    const size_t src = pixel * static_cast<size_t>(channels);
                    float value = 0.0f;
                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        value += static_cast<float>(data[src + static_cast<size_t>(c)]) / 255.0f;
                    }
                    value /= static_cast<float>(std::min(channels, 3));
                    if (options.invert_masks) {
                        value = 1.0f - value;
                    }
                    if (value >= threshold) {
                        mask.pixels[pixel] = 1;
                        ++mask.masked_pixels;
                    }
                }
            }
            free_image(data);

            if (mask.masked_pixels == 0) {
                return std::nullopt;
            }
            return mask;
        }

        void fillProjectionHoles(std::vector<glm::vec3>& color,
                                 std::vector<uint8_t>& filled,
                                 const int width,
                                 const int height,
                                 const int iterations) {
            const auto index = [width](const int x, const int y) {
                return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            };

            for (int iter = 0; iter < iterations; ++iter) {
                std::vector<glm::vec3> next_color = color;
                std::vector<uint8_t> next_filled = filled;
                bool changed = false;

                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const size_t dst = index(x, y);
                        if (filled[dst]) {
                            continue;
                        }

                        glm::vec3 sum(0.0f);
                        int count = 0;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int sy = y + dy;
                            if (sy < 0 || sy >= height) {
                                continue;
                            }
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0) {
                                    continue;
                                }
                                int sx = x + dx;
                                if (sx < 0) {
                                    sx += width;
                                } else if (sx >= width) {
                                    sx -= width;
                                }
                                const size_t src = index(sx, sy);
                                if (!filled[src]) {
                                    continue;
                                }
                                sum += color[src];
                                ++count;
                            }
                        }

                        if (count > 0) {
                            next_color[dst] = sum / static_cast<float>(count);
                            next_filled[dst] = 1;
                            changed = true;
                        }
                    }
                }

                color.swap(next_color);
                filled.swap(next_filled);
                if (!changed) {
                    break;
                }
            }
        }

    } // namespace

    std::shared_ptr<MeshData> createProjectionDomeMesh(const ProjectionDomeMeshOptions& options) {
        const int lon_segments = clampSegmentCount(options.longitude_segments, 3);
        const int lat_segments = clampSegmentCount(options.latitude_segments, 1);
        const int columns = lon_segments + 1;
        const int rows = lat_segments + 1;
        const size_t vertex_count = static_cast<size_t>(columns) * static_cast<size_t>(rows);

        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> texcoords;
        vertices.reserve(vertex_count * 3u);
        normals.reserve(vertex_count * 3u);
        texcoords.reserve(vertex_count * 2u);

        for (int y = 0; y <= lat_segments; ++y) {
            const float v = static_cast<float>(y) / static_cast<float>(lat_segments);
            for (int x = 0; x <= lon_segments; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(lon_segments);
                const glm::vec3 p = pointOnUnitDome(u, v);
                const glm::vec3 inward_normal = -glm::normalize(p);
                vertices.insert(vertices.end(), {p.x, p.y, p.z});
                normals.insert(normals.end(), {inward_normal.x, inward_normal.y, inward_normal.z});
                texcoords.insert(texcoords.end(), {u, 1.0f - v});
            }
        }

        std::vector<int> indices;
        indices.reserve(static_cast<size_t>(lon_segments) * static_cast<size_t>(lat_segments) * (options.double_sided ? 12u : 6u));
        const auto add_tri = [&indices](const int a, const int b, const int c) {
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(c);
        };

        for (int y = 0; y < lat_segments; ++y) {
            for (int x = 0; x < lon_segments; ++x) {
                const int a = y * columns + x;
                const int b = a + 1;
                const int c = (y + 1) * columns + x;
                const int d = c + 1;

                add_tri(a, b, c);
                add_tri(b, d, c);
                if (options.double_sided) {
                    add_tri(a, c, b);
                    add_tri(b, c, d);
                }
            }
        }

        auto mesh = std::make_shared<MeshData>();
        mesh->vertices = Tensor::from_vector(vertices, {vertex_count, size_t{3}}, Device::CPU);
        mesh->normals = Tensor::from_vector(normals, {vertex_count, size_t{3}}, Device::CPU);
        mesh->texcoords = Tensor::from_vector(texcoords, {vertex_count, size_t{2}}, Device::CPU);
        mesh->indices = Tensor::from_vector(indices, {indices.size() / 3u, size_t{3}}, Device::CPU);

        Material material;
        material.name = "Projection Dome Material";
        material.base_color = glm::vec4(1.0f, 1.0f, 1.0f, 0.58f);
        material.emissive = glm::vec3(0.08f);
        material.roughness = 1.0f;
        material.albedo_tex = 1;
        material.double_sided = true;
        mesh->materials.push_back(material);
        mesh->submeshes.push_back(Submesh{
            .start_index = 0,
            .index_count = indices.size(),
            .material_index = 0,
        });

        TextureImage texture;
        texture.width = std::max(2, options.placeholder_texture_width);
        texture.height = std::max(2, options.placeholder_texture_height);
        texture.channels = 3;
        texture.pixels = makePlaceholderTexture(texture.width, texture.height);
        mesh->texture_images.push_back(std::move(texture));
        return mesh;
    }

    ProjectionDomePlacement estimateProjectionDomePlacement(const Scene& scene) {
        ProjectionDomePlacement placement;
        placement.parent_id = findDatasetParent(scene);
        if (const auto center = tensorVec3(scene.getSceneCenter())) {
            placement.center = *center;
        }

        bool has_extent = false;
        glm::vec3 bounds_min(std::numeric_limits<float>::max());
        glm::vec3 bounds_max(std::numeric_limits<float>::lowest());
        if (placement.parent_id != NULL_NODE && scene.getNodeBounds(placement.parent_id, bounds_min, bounds_max)) {
            has_extent = true;
            if (!tensorVec3(scene.getSceneCenter())) {
                placement.center = (bounds_min + bounds_max) * 0.5f;
            }
        }

        float radius = 0.0f;
        const auto include_point = [&](const glm::vec3& point) {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                radius = std::max(radius, glm::length(point - placement.center));
            }
        };

        if (has_extent) {
            for (int ix = 0; ix < 2; ++ix) {
                for (int iy = 0; iy < 2; ++iy) {
                    for (int iz = 0; iz < 2; ++iz) {
                        include_point(glm::vec3(
                            ix ? bounds_max.x : bounds_min.x,
                            iy ? bounds_max.y : bounds_min.y,
                            iz ? bounds_max.z : bounds_min.z));
                    }
                }
            }
        }

        for (const auto* node : scene.getNodes()) {
            if (!node || node->type != NodeType::CAMERA || !node->camera) {
                continue;
            }
            if (const auto pose = readCameraPose(*node->camera)) {
                include_point(toParentLocal(scene, placement.parent_id, cameraWorldPosition(scene, *node, *pose)));
            }
        }

        placement.radius = std::max(radius * 1.25f, 1.0f);
        return placement;
    }

    std::optional<NodeId> findProjectionDome(const Scene& scene, const std::string_view node_name) {
        if (node_name.empty()) {
            return std::nullopt;
        }
        if (const auto* node = scene.getNode(std::string(node_name)); node && node->type == NodeType::MESH && node->mesh) {
            return node->id;
        }
        return std::nullopt;
    }

    std::expected<NodeId, std::string> ensureProjectionDome(Scene& scene,
                                                            const ProjectionDomeMeshOptions& mesh_options,
                                                            const ProjectionDomePlacement& placement) {
        const std::string node_name(kProjectionDomeNodeName);
        if (const auto* existing = scene.getNode(node_name)) {
            if (existing->type != NodeType::MESH || !existing->mesh) {
                return std::unexpected("A non-mesh node named '" + node_name + "' already exists");
            }
            return existing->id;
        }

        const NodeId node_id = scene.addMesh(node_name, createProjectionDomeMesh(mesh_options), placement.parent_id);
        if (node_id == NULL_NODE) {
            return std::unexpected("Failed to add projection dome mesh to the scene");
        }

        const glm::mat4 transform =
            glm::translate(glm::mat4(1.0f), placement.center) *
            glm::scale(glm::mat4(1.0f), glm::vec3(std::max(placement.radius, 1e-4f)));

        if (auto* node = scene.getNodeById(node_id)) {
            scene.setNodeTransform(node->name, transform);
        }
        return node_id;
    }

    std::expected<ProjectionDomeBakeResult, std::string>
    bakeProjectionDomeTexture(Scene& scene, const ProjectionDomeBakeOptions& options) {
        if (options.texture_width <= 1 || options.texture_height <= 1) {
            return std::unexpected("Projection dome texture dimensions must be larger than 1x1");
        }

        auto* node = scene.getMutableNode(options.node_name);
        if (!node || node->type != NodeType::MESH || !node->mesh) {
            return std::unexpected("Projection dome mesh node not found: " + options.node_name);
        }

        auto& mesh = *node->mesh;
        const int width = options.texture_width;
        const int height = options.texture_height;
        const size_t texel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

        std::vector<ProjectionSample> best_samples(texel_count);
        std::vector<glm::vec3> world_points(texel_count);
        std::vector<glm::vec3> inward_normals(texel_count);

        const glm::mat4 dome_world = scene.getWorldTransform(node->id);
        const glm::mat3 normal_matrix = glm::transpose(glm::inverse(glm::mat3(dome_world)));
        for (int y = 0; y < height; ++y) {
            const float v = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
            for (int x = 0; x < width; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
                const glm::vec3 local_point = pointOnUnitDome(u, v);
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                world_points[idx] = glm::vec3(dome_world * glm::vec4(local_point, 1.0f));
                inward_normals[idx] = glm::normalize(normal_matrix * (-glm::normalize(local_point)));
            }
        }

        ProjectionDomeBakeResult result{
            .width = width,
            .height = height,
        };

        for (const auto* camera_node : scene.getNodes()) {
            if (!camera_node || camera_node->type != NodeType::CAMERA || !camera_node->camera) {
                continue;
            }
            if (options.active_cameras_only && !camera_node->training_enabled) {
                continue;
            }
            ++result.cameras_considered;

            const auto pose = readCameraPose(*camera_node->camera);
            if (!pose) {
                continue;
            }

            const auto image = loadCameraImage(*camera_node->camera, options);
            if (!image) {
                continue;
            }
            ++result.cameras_used;
            const auto mask = loadCameraMask(*camera_node->camera, options);
            if (mask) {
                result.masked_pixels += mask->masked_pixels;
            }

            const glm::mat4 camera_scene = scene.getWorldTransform(camera_node->id);
            const glm::mat4 inv_camera_scene = glm::inverse(camera_scene);
            const glm::vec3 camera_world = glm::vec3(camera_scene * glm::vec4(pose->local_center, 1.0f));

            for (size_t idx = 0; idx < texel_count; ++idx) {
                const glm::vec3 view_dir = camera_world - world_points[idx];
                const float view_len = glm::length(view_dir);
                if (view_len <= 1e-6f) {
                    continue;
                }
                const float facing = std::max(glm::dot(inward_normals[idx], view_dir / view_len), 0.0f);
                if (facing <= 1e-4f) {
                    continue;
                }

                const glm::vec3 camera_data_point = glm::vec3(inv_camera_scene * glm::vec4(world_points[idx], 1.0f));
                float px = 0.0f;
                float py = 0.0f;
                bool wrap_x = false;
                if (!projectToImage(*camera_node->camera, *pose, camera_data_point,
                                    image->width, image->height, px, py, wrap_x)) {
                    continue;
                }
                if (mask) {
                    const float mask_scale_x = static_cast<float>(mask->width) / static_cast<float>(image->width);
                    const float mask_scale_y = static_cast<float>(mask->height) / static_cast<float>(image->height);
                    const int mask_x = std::clamp(static_cast<int>(std::lround(px * mask_scale_x)), 0, mask->width - 1);
                    const int mask_y = std::clamp(static_cast<int>(std::lround(py * mask_scale_y)), 0, mask->height - 1);
                    const size_t mask_idx =
                        static_cast<size_t>(mask_y) * static_cast<size_t>(mask->width) + static_cast<size_t>(mask_x);
                    if (mask->pixels[mask_idx]) {
                        ++result.samples_rejected_by_mask;
                        continue;
                    }
                }

                const float feather = edgeWeight(px, py, image->width, image->height, options.edge_falloff, wrap_x);
                // Pick one dominant source view per texel. Averaging all cameras makes
                // foreground parallax show up as repeated ghost images on the dome.
                const float score = facing * facing * facing * facing * feather * feather;
                if (score <= 1e-6f || score <= best_samples[idx].score) {
                    continue;
                }

                best_samples[idx].score = score;
                best_samples[idx].color = bilinearSample(image->pixels, image->width, image->height, px, py, wrap_x);
            }
        }

        std::vector<glm::vec3> resolved(texel_count, glm::vec3(0.0f));
        std::vector<uint8_t> filled(texel_count, 0);
        for (size_t idx = 0; idx < texel_count; ++idx) {
            if (best_samples[idx].score <= 1e-8f) {
                continue;
            }
            resolved[idx] = best_samples[idx].color;
            filled[idx] = 1;
            ++result.texels_projected;
        }

        if (result.cameras_used == 0) {
            return std::unexpected("No usable camera images were found for projection dome baking");
        }
        if (result.texels_projected == 0) {
            return std::unexpected("No dome texels were visible to the usable camera images");
        }

        fillProjectionHoles(resolved, filled, width, height, std::max(0, options.hole_fill_iterations));

        std::vector<uint8_t> output(texel_count * 3u);
        for (size_t idx = 0; idx < texel_count; ++idx) {
            if (filled[idx]) {
                ++result.texels_after_fill;
            }
            const glm::vec3 color = filled[idx] ? resolved[idx] : glm::vec3(0.0f);
            output[idx * 3u + 0] = toByte(color.r);
            output[idx * 3u + 1] = toByte(color.g);
            output[idx * 3u + 2] = toByte(color.b);
        }

        TextureImage texture;
        texture.pixels = std::move(output);
        texture.width = width;
        texture.height = height;
        texture.channels = 3;
        if (mesh.texture_images.empty()) {
            mesh.texture_images.push_back(std::move(texture));
        } else {
            mesh.texture_images[0] = std::move(texture);
        }

        if (mesh.materials.empty()) {
            Material material;
            material.name = "Projection Dome Material";
            material.base_color = glm::vec4(1.0f, 1.0f, 1.0f, 0.58f);
            material.albedo_tex = 1;
            material.roughness = 1.0f;
            material.double_sided = true;
            mesh.materials.push_back(std::move(material));
        } else {
            mesh.materials[0].albedo_tex = 1;
            mesh.materials[0].base_color.a = 0.58f;
        }
        if (mesh.submeshes.empty()) {
            mesh.submeshes.push_back(Submesh{
                .start_index = 0,
                .index_count = static_cast<size_t>(mesh.face_count()) * 3u,
                .material_index = 0,
            });
        }

        mesh.mark_dirty();
        scene.notifyMutation(Scene::MutationType::MODEL_CHANGED);
        return result;
    }

    std::expected<ProjectionDomeSkyCubemapResult, std::string>
    prepareProjectionDomeSkyCubemap(Scene& scene, const ProjectionDomeSkyCubemapOptions& options) {
        if (options.face_size < 16) {
            return std::unexpected("Sky cubemap face size must be at least 16 pixels");
        }
        if (options.output_dir.empty()) {
            return std::unexpected("Sky cubemap output directory is empty");
        }

        auto* node = scene.getMutableNode(options.node_name);
        if (!node || node->type != NodeType::MESH || !node->mesh) {
            return std::unexpected("Projection dome mesh node not found: " + options.node_name);
        }
        if (node->mesh->texture_images.empty()) {
            return std::unexpected("Projection dome has no texture to export as a sky cubemap");
        }

        const TextureImage& texture = node->mesh->texture_images.front();
        if (texture.width <= 1 || texture.height <= 1 || texture.channels <= 0 || texture.pixels.empty()) {
            return std::unexpected("Projection dome texture is empty");
        }

        std::error_code ec;
        std::filesystem::create_directories(options.output_dir, ec);
        if (ec) {
            return std::unexpected("Failed to create sky cubemap directory '" +
                                   path_to_utf8(options.output_dir) + "': " + ec.message());
        }

        ProjectionDomeSkyCubemapResult result;
        result.output_dir = options.output_dir;
        result.face_size = options.face_size;
        result.faces.reserve(kCubemapFaces.size());

        for (const auto& face : kCubemapFaces) {
            ProjectionDomeSkyCubemapFace face_result;
            face_result.id = std::string(face.id);
            face_result.label = std::string(face.label);
            face_result.preview_path = previewPath(options.output_dir, face.id);
            face_result.mask_path = maskPath(options.output_dir, face.id);
            face_result.overlay_path = overlayPath(options.output_dir, face.id);

            if (options.overwrite_preview || !std::filesystem::exists(face_result.preview_path)) {
                std::vector<uint8_t> preview(static_cast<size_t>(options.face_size) *
                                             static_cast<size_t>(options.face_size) * 4u,
                                             0);

                for (int y = 0; y < options.face_size; ++y) {
                    const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) /
                                             static_cast<float>(options.face_size);
                    for (int x = 0; x < options.face_size; ++x) {
                        const float a = 2.0f * (static_cast<float>(x) + 0.5f) /
                                            static_cast<float>(options.face_size) -
                                        1.0f;
                        const glm::vec3 dir = cubemapDirection(face.id, a, b);
                        const size_t idx = static_cast<size_t>(y) *
                                               static_cast<size_t>(options.face_size) +
                                           static_cast<size_t>(x);
                        if (!isSkyDirection(dir)) {
                            preview[idx * 4u + 0] = 20;
                            preview[idx * 4u + 1] = 22;
                            preview[idx * 4u + 2] = 26;
                            preview[idx * 4u + 3] = 95;
                            continue;
                        }

                        const glm::vec3 color = sampleDomeTexture(texture, dir);
                        preview[idx * 4u + 0] = toByte(color.r);
                        preview[idx * 4u + 1] = toByte(color.g);
                        preview[idx * 4u + 2] = toByte(color.b);
                        preview[idx * 4u + 3] = 255;
                        ++face_result.valid_pixels;
                    }
                }

                if (!saveRawImage(face_result.preview_path, preview, options.face_size, options.face_size, 4)) {
                    return std::unexpected("Failed to save sky cubemap preview face: " +
                                           path_to_utf8(face_result.preview_path));
                }
            } else {
                // We still need a validity estimate for UI summaries. Recompute from directions;
                // it is cheap and independent of the saved preview.
                for (int y = 0; y < options.face_size; ++y) {
                    const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) /
                                             static_cast<float>(options.face_size);
                    for (int x = 0; x < options.face_size; ++x) {
                        const float a = 2.0f * (static_cast<float>(x) + 0.5f) /
                                            static_cast<float>(options.face_size) -
                                        1.0f;
                        if (isSkyDirection(cubemapDirection(face.id, a, b))) {
                            ++face_result.valid_pixels;
                        }
                    }
                }
            }

            std::vector<uint8_t> mask;
            if (options.reset_mask || !std::filesystem::exists(face_result.mask_path)) {
                mask.assign(static_cast<size_t>(options.face_size) *
                                static_cast<size_t>(options.face_size),
                            0);
            } else {
                mask = loadMaskPixels(face_result.mask_path, options.face_size);
            }
            face_result.marked_pixels = countMarkedPixels(mask);

            if (!writeMaskAndOverlay(face_result.mask_path, face_result.overlay_path,
                                     mask, options.face_size)) {
                return std::unexpected("Failed to save sky mask face: " +
                                       path_to_utf8(face_result.mask_path));
            }

            result.faces.push_back(std::move(face_result));
        }

        return result;
    }

    std::expected<ProjectionDomeSkyMaskPaintResult, std::string>
    paintProjectionDomeSkyMask(const std::filesystem::path& mask_path,
                               const std::filesystem::path& overlay_path,
                               int face_size,
                               int x,
                               int y,
                               int radius,
                               const bool erase) {
        if (face_size < 16) {
            return std::unexpected("Sky cubemap face size must be at least 16 pixels");
        }
        radius = std::clamp(radius, 1, face_size);
        x = std::clamp(x, 0, face_size - 1);
        y = std::clamp(y, 0, face_size - 1);

        std::error_code ec;
        std::filesystem::create_directories(mask_path.parent_path(), ec);
        if (ec) {
            return std::unexpected("Failed to create sky mask directory '" +
                                   path_to_utf8(mask_path.parent_path()) + "': " + ec.message());
        }
        std::filesystem::create_directories(overlay_path.parent_path(), ec);
        if (ec) {
            return std::unexpected("Failed to create sky overlay directory '" +
                                   path_to_utf8(overlay_path.parent_path()) + "': " + ec.message());
        }

        auto mask = loadMaskPixels(mask_path, face_size);
        const uint8_t value = erase ? 0 : 255;
        int changed = 0;
        const int radius_sq = radius * radius;
        for (int py = std::max(0, y - radius); py <= std::min(face_size - 1, y + radius); ++py) {
            for (int px = std::max(0, x - radius); px <= std::min(face_size - 1, x + radius); ++px) {
                const int dx = px - x;
                const int dy = py - y;
                if (dx * dx + dy * dy > radius_sq) {
                    continue;
                }
                const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(face_size) +
                                   static_cast<size_t>(px);
                if (mask[idx] == value) {
                    continue;
                }
                mask[idx] = value;
                ++changed;
            }
        }

        if (!writeMaskAndOverlay(mask_path, overlay_path, mask, face_size)) {
            return std::unexpected("Failed to save sky mask face: " + path_to_utf8(mask_path));
        }

        return ProjectionDomeSkyMaskPaintResult{
            .marked_pixels = countMarkedPixels(mask),
            .changed_pixels = changed,
        };
    }

    std::expected<ProjectionDomeSkyMaskPaintResult, std::string>
    clearProjectionDomeSkyMask(const std::filesystem::path& mask_path,
                               const std::filesystem::path& overlay_path,
                               const int face_size) {
        if (face_size < 16) {
            return std::unexpected("Sky cubemap face size must be at least 16 pixels");
        }
        auto mask = loadMaskPixels(mask_path, face_size);
        const int changed = countMarkedPixels(mask);
        std::fill(mask.begin(), mask.end(), uint8_t{0});

        std::error_code ec;
        std::filesystem::create_directories(mask_path.parent_path(), ec);
        if (ec) {
            return std::unexpected("Failed to create sky mask directory '" +
                                   path_to_utf8(mask_path.parent_path()) + "': " + ec.message());
        }
        std::filesystem::create_directories(overlay_path.parent_path(), ec);
        if (ec) {
            return std::unexpected("Failed to create sky overlay directory '" +
                                   path_to_utf8(overlay_path.parent_path()) + "': " + ec.message());
        }

        if (!writeMaskAndOverlay(mask_path, overlay_path, mask, face_size)) {
            return std::unexpected("Failed to save sky mask face: " + path_to_utf8(mask_path));
        }
        return ProjectionDomeSkyMaskPaintResult{
            .marked_pixels = 0,
            .changed_pixels = changed,
        };
    }

    std::expected<ProjectionDomeSkyPointCloudResult, std::string>
    createProjectionDomeSkyPointCloud(const Scene& scene,
                                      const ProjectionDomeSkyPointCloudOptions& options) {
        if (options.manifest_path.empty()) {
            return std::unexpected("Sky mask manifest path is empty");
        }
        if (options.max_gaussians <= 0) {
            return ProjectionDomeSkyPointCloudResult{};
        }

        const auto* node = scene.getNode(options.node_name);
        if (!node || node->type != NodeType::MESH || !node->mesh) {
            return std::unexpected("Projection dome mesh node not found: " + options.node_name);
        }
        if (node->mesh->texture_images.empty()) {
            return std::unexpected("Projection dome has no texture for sky initialization");
        }
        if (!std::filesystem::exists(options.manifest_path)) {
            return std::unexpected("Sky mask manifest does not exist: " +
                                   path_to_utf8(options.manifest_path));
        }

        nlohmann::json manifest;
        try {
            std::ifstream file(options.manifest_path);
            if (!file) {
                return std::unexpected("Failed to open sky mask manifest: " +
                                       path_to_utf8(options.manifest_path));
            }
            file >> manifest;
        } catch (const std::exception& e) {
            return std::unexpected("Failed to parse sky mask manifest: " + std::string(e.what()));
        }

        const int face_size = manifest.value("face_size", 0);
        if (face_size < 16 || !manifest.contains("faces") || !manifest["faces"].is_object()) {
            return std::unexpected("Sky mask manifest is missing cubemap face data");
        }

        struct FaceInput {
            std::string id;
            std::filesystem::path mask_path;
            std::filesystem::path preview_path;
            std::vector<uint8_t> mask;
            std::optional<LoadedImage> preview;
            int marked_pixels = 0;
        };

        std::vector<FaceInput> faces;
        int total_marked = 0;
        const auto& faces_json = manifest["faces"];
        for (const auto& face_def : kCubemapFaces) {
            const std::string face_id(face_def.id);
            if (!faces_json.contains(face_id) || !faces_json[face_id].is_object()) {
                continue;
            }

            const auto& face_json = faces_json[face_id];
            const std::filesystem::path mask_path =
                resolveManifestPath(options.manifest_path,
                                    face_json.contains("mask") ? face_json["mask"] : nlohmann::json{});
            if (mask_path.empty()) {
                continue;
            }

            FaceInput face;
            face.id = face_id;
            face.mask_path = mask_path;
            face.preview_path =
                resolveManifestPath(options.manifest_path,
                                    face_json.contains("preview") ? face_json["preview"] : nlohmann::json{});
            face.mask = loadMaskPixels(face.mask_path, face_size);
            face.marked_pixels = countMarkedPixels(face.mask);
            total_marked += face.marked_pixels;
            if (face.marked_pixels > 0) {
                face.preview = loadRawImage(face.preview_path);
            }
            faces.push_back(std::move(face));
        }

        if (total_marked <= 0) {
            return ProjectionDomeSkyPointCloudResult{
                .marked_pixels = 0,
                .gaussian_count = 0,
            };
        }

        const int stride = std::max(
            1,
            static_cast<int>(std::ceil(std::sqrt(
                static_cast<double>(total_marked) /
                static_cast<double>(std::max(1, options.max_gaussians))))));

        const glm::mat4 dome_world = scene.getWorldTransform(node->id);
        const TextureImage& dome_texture = node->mesh->texture_images.front();

        std::vector<float> positions;
        std::vector<uint8_t> colors;
        positions.reserve(static_cast<size_t>(std::min(total_marked, options.max_gaussians)) * 3u);
        colors.reserve(static_cast<size_t>(std::min(total_marked, options.max_gaussians)) * 3u);

        for (const FaceInput& face : faces) {
            if (face.marked_pixels <= 0) {
                continue;
            }

            for (int y = 0; y < face_size; ++y) {
                if ((y % stride) != 0) {
                    continue;
                }
                const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) /
                                         static_cast<float>(face_size);
                for (int x = 0; x < face_size; ++x) {
                    if ((x % stride) != 0) {
                        continue;
                    }
                    const size_t mask_idx = static_cast<size_t>(y) *
                                                static_cast<size_t>(face_size) +
                                            static_cast<size_t>(x);
                    if (face.mask[mask_idx] < 128) {
                        continue;
                    }

                    const float a = 2.0f * (static_cast<float>(x) + 0.5f) /
                                        static_cast<float>(face_size) -
                                    1.0f;
                    const glm::vec3 dir = cubemapDirection(face.id, a, b);
                    if (!isSkyDirection(dir)) {
                        continue;
                    }

                    const glm::vec3 world_point =
                        glm::vec3(dome_world * glm::vec4(dir, 1.0f));
                    const glm::vec3 output_point =
                        glm::vec3(options.output_from_world * glm::vec4(world_point, 1.0f));
                    if (!std::isfinite(output_point.x) ||
                        !std::isfinite(output_point.y) ||
                        !std::isfinite(output_point.z)) {
                        continue;
                    }

                    const glm::vec3 color =
                        samplePreviewColor(face.preview, dome_texture, face.id,
                                           face_size, x, y, dir);
                    positions.insert(positions.end(), {output_point.x, output_point.y, output_point.z});
                    colors.push_back(toByte(color.r));
                    colors.push_back(toByte(color.g));
                    colors.push_back(toByte(color.b));
                }
            }
        }

        const size_t candidate_count = positions.size() / 3u;
        if (candidate_count == 0) {
            return ProjectionDomeSkyPointCloudResult{
                .marked_pixels = total_marked,
                .gaussian_count = 0,
            };
        }

        if (candidate_count > static_cast<size_t>(options.max_gaussians)) {
            std::vector<float> limited_positions;
            std::vector<uint8_t> limited_colors;
            const size_t target = static_cast<size_t>(options.max_gaussians);
            limited_positions.reserve(target * 3u);
            limited_colors.reserve(target * 3u);
            for (size_t i = 0; i < target; ++i) {
                const size_t src = (target == 1)
                                       ? 0
                                       : std::min(candidate_count - 1,
                                                  (i * (candidate_count - 1)) / (target - 1));
                limited_positions.insert(limited_positions.end(),
                                         positions.begin() + static_cast<std::ptrdiff_t>(src * 3u),
                                         positions.begin() + static_cast<std::ptrdiff_t>(src * 3u + 3u));
                limited_colors.insert(limited_colors.end(),
                                      colors.begin() + static_cast<std::ptrdiff_t>(src * 3u),
                                      colors.begin() + static_cast<std::ptrdiff_t>(src * 3u + 3u));
            }
            positions = std::move(limited_positions);
            colors = std::move(limited_colors);
        }

        const size_t gaussian_count = positions.size() / 3u;
        auto means = Tensor::from_vector(positions, {gaussian_count, size_t{3}}, Device::CPU);
        auto color_tensor = Tensor::empty({gaussian_count, size_t{3}}, Device::CPU, DataType::UInt8);
        std::memcpy(color_tensor.data_ptr(), colors.data(), colors.size() * sizeof(uint8_t));

        return ProjectionDomeSkyPointCloudResult{
            .point_cloud = PointCloud(std::move(means), std::move(color_tensor)),
            .marked_pixels = total_marked,
            .gaussian_count = static_cast<int>(gaussian_count),
        };
    }

} // namespace lfs::core
