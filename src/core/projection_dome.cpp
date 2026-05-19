/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/projection_dome.hpp"

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
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

        [[nodiscard]] float smoothstep01(const float edge0, const float edge1, const float value) {
            if (edge1 <= edge0) {
                return value >= edge1 ? 1.0f : 0.0f;
            }
            const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        [[nodiscard]] uint32_t hashSkyCell(const std::string_view face_id,
                                           const int cell_x,
                                           const int cell_y) {
            uint32_t h = 2166136261u;
            for (const char c : face_id) {
                h ^= static_cast<uint8_t>(c);
                h *= 16777619u;
            }
            h ^= static_cast<uint32_t>(cell_x) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= static_cast<uint32_t>(cell_y) + 0x85ebca6bu + (h << 6u) + (h >> 2u);
            h ^= h >> 16u;
            h *= 0x7feb352du;
            h ^= h >> 15u;
            h *= 0x846ca68bu;
            h ^= h >> 16u;
            return h;
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

        [[nodiscard]] std::optional<glm::mat4> readManifestMat4(const nlohmann::json& manifest,
                                                                const std::string_view key) {
            const std::string key_string(key);
            if (!manifest.contains(key_string) || !manifest[key_string].is_array() ||
                manifest[key_string].size() != 16) {
                return std::nullopt;
            }

            glm::mat4 result(1.0f);
            float* values = &result[0][0];
            for (size_t i = 0; i < 16; ++i) {
                if (!manifest[key_string][i].is_number()) {
                    return std::nullopt;
                }
                values[i] = manifest[key_string][i].get<float>();
                if (!std::isfinite(values[i])) {
                    return std::nullopt;
                }
            }
            return result;
        }

        [[nodiscard]] std::optional<glm::vec3> samplePreviewFaceColor(const std::optional<LoadedImage>& preview,
                                                                      const int face_size,
                                                                      const int x,
                                                                      const int y) {
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

            return std::nullopt;
        }

        [[nodiscard]] std::optional<glm::vec3> sampleDomeSkyColor(const TextureImage& dome_texture,
                                                                  const glm::vec3& dir) {
            if (dome_texture.width <= 1 || dome_texture.height <= 1 || dome_texture.pixels.empty()) {
                return std::nullopt;
            }
            return sampleDomeTexture(dome_texture, dir);
        }

        [[nodiscard]] float skyColorConfidence(const glm::vec3& color) {
            if (!std::isfinite(color.r) || !std::isfinite(color.g) || !std::isfinite(color.b)) {
                return 0.0f;
            }
            const float r = std::clamp(color.r, 0.0f, 1.0f);
            const float g = std::clamp(color.g, 0.0f, 1.0f);
            const float b = std::clamp(color.b, 0.0f, 1.0f);
            const float max_channel = std::max(r, std::max(g, b));
            const float min_channel = std::min(r, std::min(g, b));
            const float saturation = max_channel - min_channel;
            const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (luma < 0.08f || max_channel < 0.12f) {
                return 0.0f;
            }

            const float warm_bias = std::max(r - b, g - b);
            const float cool_or_neutral = 1.0f - smoothstep01(0.04f, 0.18f, warm_bias);
            const float bright_neutral = smoothstep01(0.40f, 0.70f, luma) *
                                         (1.0f - smoothstep01(0.12f, 0.36f, saturation)) *
                                         cool_or_neutral;
            const float blue = smoothstep01(0.02f, 0.18f, b - r) *
                               smoothstep01(0.02f, 0.16f, b - g) *
                               smoothstep01(0.34f, 0.62f, b);
            const float green_reject = smoothstep01(0.02f, 0.18f, g - std::max(r, b)) *
                                       smoothstep01(0.20f, 0.45f, g);
            const float dark_reject = 1.0f - smoothstep01(0.10f, 0.28f, luma);
            return std::clamp(std::max(bright_neutral, blue) * (1.0f - green_reject) * (1.0f - dark_reject),
                              0.0f,
                              1.0f);
        }

        [[nodiscard]] bool isUsableSkyColor(const glm::vec3& color) {
            if (!std::isfinite(color.r) || !std::isfinite(color.g) || !std::isfinite(color.b)) {
                return false;
            }
            const float luma = 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
            const float max_channel = std::max(color.r, std::max(color.g, color.b));
            return luma > 0.08f && max_channel > 0.12f && skyColorConfidence(color) > 0.12f;
        }

        [[nodiscard]] glm::vec3 clampColor(const glm::vec3& color) {
            return glm::vec3(std::clamp(color.r, 0.0f, 1.0f),
                             std::clamp(color.g, 0.0f, 1.0f),
                             std::clamp(color.b, 0.0f, 1.0f));
        }

        [[nodiscard]] glm::vec3 fallbackSkyGradient(const glm::vec3& dir) {
            const float t = std::pow(std::clamp(skyElevation01(dir), 0.0f, 1.0f), 0.55f);
            const glm::vec3 horizon(0.82f, 0.88f, 0.96f);
            const glm::vec3 zenith(0.45f, 0.62f, 0.90f);
            return glm::mix(horizon, zenith, t);
        }

        [[nodiscard]] glm::vec3 fallbackSkyColor(const glm::vec3& dir,
                                                 const std::optional<glm::vec3>& dominant_color = std::nullopt) {
            const glm::vec3 gradient = fallbackSkyGradient(dir);
            if (!dominant_color) {
                return gradient;
            }
            return glm::mix(gradient, clampColor(*dominant_color), 0.72f);
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
                                 const int iterations,
                                 const std::vector<uint8_t>* domain = nullptr) {
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
                        if (domain && (dst >= domain->size() || (*domain)[dst] == 0)) {
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
                                if (domain && (src >= domain->size() || (*domain)[src] == 0)) {
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

        [[nodiscard]] float maskNeighborhoodCoverage(const std::vector<uint8_t>& domain,
                                                     const int width,
                                                     const int height,
                                                     const int x,
                                                     const int y,
                                                     const int radius) {
            int covered = 0;
            int total = 0;
            const auto index = [width](const int sx, const int sy) {
                return static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
            };

            for (int dy = -radius; dy <= radius; ++dy) {
                const int sy = y + dy;
                if (sy < 0 || sy >= height) {
                    total += radius * 2 + 1;
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = x + dx;
                    if (sx < 0) {
                        sx += width;
                    } else if (sx >= width) {
                        sx -= width;
                    }
                    ++total;
                    const size_t idx = index(sx, sy);
                    if (idx < domain.size() && domain[idx] != 0) {
                        ++covered;
                    }
                }
            }

            return total > 0 ? static_cast<float>(covered) / static_cast<float>(total) : 0.0f;
        }

        struct SkyFaceColorMap {
            std::vector<glm::vec3> color;
            std::vector<uint8_t> filled;
            int seeded_pixels = 0;
            int rejected_pixels = 0;
            int fallback_pixels = 0;
        };

        struct SkyColorStats {
            glm::vec3 weighted_sum{0.0f};
            glm::vec3 weighted_sq_sum{0.0f};
            float weight_sum = 0.0f;
            int accepted_samples = 0;
            int rejected_samples = 0;
        };

        void addSkyColorSample(SkyColorStats& stats, const glm::vec3& color) {
            const float confidence = skyColorConfidence(color);
            if (confidence <= 0.12f || !isUsableSkyColor(color)) {
                ++stats.rejected_samples;
                return;
            }
            const float weight = confidence * confidence;
            const glm::vec3 clamped = clampColor(color);
            stats.weighted_sum += clamped * weight;
            stats.weighted_sq_sum += clamped * clamped * weight;
            stats.weight_sum += weight;
            ++stats.accepted_samples;
        }

        void accumulateMaskedSkyColorSamples(SkyColorStats& stats,
                                             const std::optional<LoadedImage>& preview,
                                             const TextureImage& dome_texture,
                                             const std::string_view face_id,
                                             const std::vector<uint8_t>& mask,
                                             const int face_size) {
            for (int y = 0; y < face_size; ++y) {
                const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(face_size);
                for (int x = 0; x < face_size; ++x) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(face_size) + static_cast<size_t>(x);
                    if (idx >= mask.size() || mask[idx] < 128) {
                        continue;
                    }
                    const float a = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(face_size) - 1.0f;
                    const glm::vec3 dir = cubemapDirection(face_id, a, b);
                    if (!isSkyDirection(dir)) {
                        continue;
                    }

                    if (const auto preview_color = samplePreviewFaceColor(preview, face_size, x, y)) {
                        addSkyColorSample(stats, *preview_color);
                    }
                    if (const auto dome_color = sampleDomeSkyColor(dome_texture, dir)) {
                        addSkyColorSample(stats, *dome_color);
                    }
                }
            }
        }

        [[nodiscard]] std::optional<glm::vec3> dominantSkyColor(const SkyColorStats& stats) {
            if (stats.weight_sum <= 1.0e-6f || stats.accepted_samples <= 0) {
                return std::nullopt;
            }
            const glm::vec3 color = clampColor(stats.weighted_sum / stats.weight_sum);
            if (!isUsableSkyColor(color)) {
                return std::nullopt;
            }
            return color;
        }

        [[nodiscard]] float dominantSkyColorRadius(const SkyColorStats& stats, const glm::vec3& mean) {
            if (stats.weight_sum <= 1.0e-6f || stats.accepted_samples <= 0) {
                return 0.0f;
            }
            const glm::vec3 second_moment = stats.weighted_sq_sum / stats.weight_sum;
            const glm::vec3 variance = glm::max(second_moment - mean * mean, glm::vec3(0.0f));
            const float rms_radius = std::sqrt(std::max(0.0f, variance.r + variance.g + variance.b));
            return std::clamp(2.5f * rms_radius + 0.06f, 0.12f, 0.42f);
        }

        [[nodiscard]] SkyFaceColorMap buildSkyFaceColorMap(
            const std::optional<LoadedImage>& preview,
            const TextureImage& dome_texture,
            const std::string_view face_id,
            const std::vector<uint8_t>& mask,
            const int face_size,
            const std::optional<glm::vec3>& dominant_sky_color) {
            const size_t pixel_count = static_cast<size_t>(face_size) * static_cast<size_t>(face_size);
            SkyFaceColorMap map;
            map.color.assign(pixel_count, glm::vec3(0.0f));
            map.filled.assign(pixel_count, 0);

            std::vector<uint8_t> domain(pixel_count, 0);
            for (int y = 0; y < face_size; ++y) {
                const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(face_size);
                for (int x = 0; x < face_size; ++x) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(face_size) + static_cast<size_t>(x);
                    if (idx >= mask.size() || mask[idx] < 128) {
                        continue;
                    }
                    const float a = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(face_size) - 1.0f;
                    const glm::vec3 dir = cubemapDirection(face_id, a, b);
                    if (!isSkyDirection(dir)) {
                        continue;
                    }

                    domain[idx] = 1;
                }
            }

            for (int y = 0; y < face_size; ++y) {
                const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(face_size);
                for (int x = 0; x < face_size; ++x) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(face_size) + static_cast<size_t>(x);
                    if (idx >= domain.size() || domain[idx] == 0) {
                        continue;
                    }
                    const float a = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(face_size) - 1.0f;
                    const glm::vec3 dir = cubemapDirection(face_id, a, b);

                    float best_confidence = 0.0f;
                    glm::vec3 best_color(0.0f);
                    bool has_candidate = false;

                    if (const auto preview_color = samplePreviewFaceColor(preview, face_size, x, y)) {
                        const float confidence = skyColorConfidence(*preview_color);
                        if (confidence > best_confidence) {
                            best_confidence = confidence;
                            best_color = *preview_color;
                            has_candidate = true;
                        }
                    }
                    if (const auto dome_color = sampleDomeSkyColor(dome_texture, dir)) {
                        const float confidence = skyColorConfidence(*dome_color);
                        if (confidence > best_confidence) {
                            best_confidence = confidence;
                            best_color = *dome_color;
                            has_candidate = true;
                        }
                    }

                    const float interior = maskNeighborhoodCoverage(domain, face_size, face_size, x, y, 2);
                    const float min_confidence = interior >= 0.82f ? 0.18f : 0.55f;
                    if (!has_candidate || best_confidence < min_confidence || !isUsableSkyColor(best_color)) {
                        ++map.rejected_pixels;
                        continue;
                    }

                    const float sample_weight = smoothstep01(min_confidence, 0.78f, best_confidence);
                    const glm::vec3 color = glm::mix(fallbackSkyColor(dir, dominant_sky_color), best_color, sample_weight);

                    map.color[idx] = color;
                    map.filled[idx] = 1;
                    ++map.seeded_pixels;
                }
            }

            fillProjectionHoles(map.color, map.filled, face_size, face_size, std::max(16, face_size / 2), &domain);

            for (int y = 0; y < face_size; ++y) {
                const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(face_size);
                for (int x = 0; x < face_size; ++x) {
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(face_size) + static_cast<size_t>(x);
                    if (idx >= domain.size() || domain[idx] == 0 || map.filled[idx]) {
                        continue;
                    }
                    const float a = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(face_size) - 1.0f;
                    const glm::vec3 dir = cubemapDirection(face_id, a, b);
                    map.color[idx] = fallbackSkyColor(dir, dominant_sky_color);
                    map.filled[idx] = 1;
                    ++map.fallback_pixels;
                }
            }

            return map;
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

        std::vector<glm::vec3> camera_points;
        for (const auto* node : scene.getNodes()) {
            if (!node || node->type != NodeType::CAMERA || !node->camera) {
                continue;
            }
            if (const auto pose = readCameraPose(*node->camera)) {
                camera_points.push_back(toParentLocal(scene, placement.parent_id,
                                                      cameraWorldPosition(scene, *node, *pose)));
            }
        }

        if (!camera_points.empty()) {
            glm::vec3 camera_center(0.0f);
            for (const glm::vec3& point : camera_points) {
                camera_center += point;
            }
            camera_center /= static_cast<float>(camera_points.size());
            placement.center = camera_center;

            float camera_radius = 0.0f;
            for (const glm::vec3& point : camera_points) {
                camera_radius = std::max(camera_radius, glm::length(point - placement.center));
            }

            float camera_diameter = camera_radius * 2.0f;
            if (camera_points.size() <= 2048u) {
                for (size_t i = 0; i < camera_points.size(); ++i) {
                    for (size_t j = i + 1u; j < camera_points.size(); ++j) {
                        camera_diameter = std::max(camera_diameter,
                                                   glm::length(camera_points[i] - camera_points[j]));
                    }
                }
            }

            // Use the camera navigation volume for sky placement. Sparse COLMAP point clouds
            // often include sky/foreground outliers, and those should not push the dome to
            // scene-bound distances.
            placement.radius = std::max(camera_diameter * 10.0f, 1.0f);
            return placement;
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
        result.dome_world = scene.getWorldTransform(node->id);
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

        const glm::mat4 dome_world = readManifestMat4(manifest, "dome_world")
                                         .value_or(scene.getWorldTransform(node->id));
        const TextureImage& dome_texture = node->mesh->texture_images.front();

        struct FaceInput {
            std::string id;
            std::filesystem::path mask_path;
            std::filesystem::path preview_path;
            std::vector<uint8_t> mask;
            std::optional<LoadedImage> preview;
            SkyFaceColorMap color_map;
            int marked_pixels = 0;
        };

        std::vector<FaceInput> faces;
        SkyColorStats masked_sky_stats;
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
                accumulateMaskedSkyColorSamples(masked_sky_stats,
                                                face.preview,
                                                dome_texture,
                                                face.id,
                                                face.mask,
                                                face_size);
            }
            faces.push_back(std::move(face));
        }

        if (total_marked <= 0) {
            return ProjectionDomeSkyPointCloudResult{
                .marked_pixels = 0,
                .gaussian_count = 0,
            };
        }

        const std::optional<glm::vec3> dominant_sky_color = dominantSkyColor(masked_sky_stats);
        const float dominant_sky_radius =
            dominant_sky_color ? dominantSkyColorRadius(masked_sky_stats, *dominant_sky_color) : 0.0f;
        if (dominant_sky_color) {
            LOG_INFO("Sky initialization dominant masked sky color rgb=({:.2f}, {:.2f}, {:.2f}), radius {:.3f} from {} confident samples ({} rejected)",
                     dominant_sky_color->r,
                     dominant_sky_color->g,
                     dominant_sky_color->b,
                     dominant_sky_radius,
                     masked_sky_stats.accepted_samples,
                     masked_sky_stats.rejected_samples);
        } else {
            LOG_INFO("Sky initialization found no confident masked sky color samples ({} rejected); using directional fallback",
                     masked_sky_stats.rejected_samples);
        }

        for (FaceInput& face : faces) {
            if (face.marked_pixels <= 0) {
                continue;
            }
            face.color_map = buildSkyFaceColorMap(
                face.preview,
                dome_texture,
                face.id,
                face.mask,
                face_size,
                dominant_sky_color);
            if (face.color_map.rejected_pixels > 0 || face.color_map.fallback_pixels > 0) {
                LOG_INFO("Sky initialization propagated face {}: {} valid seeds, {} rejected samples, {} fallback pixels",
                         face.id,
                         face.color_map.seeded_pixels,
                         face.color_map.rejected_pixels,
                         face.color_map.fallback_pixels);
            }
        }

        const double marked_to_target =
            static_cast<double>(total_marked) /
            static_cast<double>(std::max(1, options.max_gaussians));
        const int stride = std::max(
            1,
            static_cast<int>(std::floor(std::sqrt(std::max(1.0, marked_to_target)))));
        LOG_INFO("Sky initialization sampling {} marked pixels toward {} gaussians with jittered stride {}",
                 total_marked,
                 options.max_gaussians,
                 stride);

        std::vector<float> positions;
        std::vector<uint8_t> colors;
        const size_t reserve_count = std::min(
            static_cast<size_t>(total_marked),
            static_cast<size_t>(std::max(1, options.max_gaussians)) * 2u);
        positions.reserve(reserve_count * 3u);
        colors.reserve(reserve_count * 3u);

        for (const FaceInput& face : faces) {
            if (face.marked_pixels <= 0) {
                continue;
            }

            for (int y0 = 0; y0 < face_size; y0 += stride) {
                const int y_span = std::min(stride, face_size - y0);
                for (int x0 = 0; x0 < face_size; x0 += stride) {
                    const int x_span = std::min(stride, face_size - x0);
                    const int block_pixels = std::max(1, x_span * y_span);
                    const uint32_t seed = hashSkyCell(face.id, x0 / stride, y0 / stride);
                    const int start = static_cast<int>(seed % static_cast<uint32_t>(block_pixels));

                    bool emitted = false;
                    for (int attempt = 0; attempt < block_pixels && !emitted; ++attempt) {
                        const int local = (start + attempt) % block_pixels;
                        const int x = x0 + (local % x_span);
                        const int y = y0 + (local / x_span);
                        const size_t mask_idx = static_cast<size_t>(y) *
                                                    static_cast<size_t>(face_size) +
                                                static_cast<size_t>(x);
                        if (face.mask[mask_idx] < 128) {
                            continue;
                        }

                        const float a = 2.0f * (static_cast<float>(x) + 0.5f) /
                                            static_cast<float>(face_size) -
                                        1.0f;
                        const float b = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) /
                                               static_cast<float>(face_size);
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

                        const glm::vec3 color = (mask_idx < face.color_map.color.size() && face.color_map.filled[mask_idx])
                                                    ? face.color_map.color[mask_idx]
                                                    : fallbackSkyColor(dir, dominant_sky_color);
                        positions.insert(positions.end(), {output_point.x, output_point.y, output_point.z});
                        colors.push_back(toByte(color.r));
                        colors.push_back(toByte(color.g));
                        colors.push_back(toByte(color.b));
                        emitted = true;
                    }
                }
            }
        }

        const size_t candidate_count = positions.size() / 3u;
        if (candidate_count == 0) {
            return ProjectionDomeSkyPointCloudResult{
                .marked_pixels = total_marked,
                .gaussian_count = 0,
                .dominant_color = dominant_sky_color,
                .dominant_color_radius = dominant_sky_radius,
                .dominant_color_samples = masked_sky_stats.accepted_samples,
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
            .dominant_color = dominant_sky_color,
            .dominant_color_radius = dominant_sky_radius,
            .dominant_color_samples = masked_sky_stats.accepted_samples,
        };
    }

} // namespace lfs::core
