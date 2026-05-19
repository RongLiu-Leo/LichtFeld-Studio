/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/projection_dome.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "dataset.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <format>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lfs::training {

    namespace {
        std::shared_ptr<lfs::core::PointCloud> createRandomPointCloud() {
            constexpr size_t N = 10000;
            auto positions = lfs::core::Tensor::rand({N, 3}, lfs::core::Device::CPU) * 2.0f - 1.0f;
            auto colors = lfs::core::Tensor::randint({N, 3}, 0, 256, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            return std::make_shared<lfs::core::PointCloud>(positions, colors);
        }

        lfs::io::CentralizeDataset parse_centralize(const std::string& s) {
            if (s == "by_pointcloud")
                return lfs::io::CentralizeDataset::ByPointCloud;
            if (s == "by_cameras")
                return lfs::io::CentralizeDataset::ByCameras;
            return lfs::io::CentralizeDataset::Off;
        }

        void applyTrainingSHDegree(lfs::core::SplatData& splat, const int target_degree) {
            const int before = splat.get_max_sh_degree();
            if (splat.set_sh_degree(target_degree)) {
                LOG_INFO("Adjusted training model SH degree: {} -> {}", before, splat.get_max_sh_degree());
            }
        }

        lfs::core::PointCloud selectPointCloudRows(const lfs::core::PointCloud& point_cloud,
                                                   const size_t target_count) {
            const size_t source_count = static_cast<size_t>(point_cloud.size());
            if (target_count >= source_count) {
                return lfs::core::PointCloud(point_cloud.means.cpu(), point_cloud.colors.cpu());
            }
            if (target_count == 0 || source_count == 0) {
                return {};
            }

            std::vector<int> indices;
            indices.reserve(target_count);
            for (size_t i = 0; i < target_count; ++i) {
                const size_t src = (target_count == 1)
                                       ? 0
                                       : std::min(source_count - 1,
                                                  (i * (source_count - 1)) / (target_count - 1));
                indices.push_back(static_cast<int>(src));
            }

            auto index_tensor = lfs::core::Tensor::from_vector(
                                    indices, {target_count}, lfs::core::Device::CPU)
                                    .to(lfs::core::DataType::Int64);
            return lfs::core::PointCloud(
                point_cloud.means.cpu().index_select(0, index_tensor).contiguous(),
                point_cloud.colors.cpu().index_select(0, index_tensor).contiguous());
        }

        void prependPointCloud(lfs::core::PointCloud& target,
                               const lfs::core::PointCloud& prefix) {
            if (prefix.size() <= 0) {
                return;
            }
            if (target.size() <= 0) {
                target = lfs::core::PointCloud(prefix.means.cpu(), prefix.colors.cpu());
                return;
            }
            target.means = lfs::core::Tensor::cat({prefix.means.cpu(), target.means.cpu()}, 0);
            target.colors = lfs::core::Tensor::cat({prefix.colors.cpu(), target.colors.cpu()}, 0);
        }

        void appendPointCloud(lfs::core::PointCloud& target,
                              const lfs::core::PointCloud& suffix) {
            if (suffix.size() <= 0) {
                return;
            }
            if (target.size() <= 0) {
                target = lfs::core::PointCloud(suffix.means.cpu(), suffix.colors.cpu());
                return;
            }
            target.means = lfs::core::Tensor::cat({target.means.cpu(), suffix.means.cpu()}, 0);
            target.colors = lfs::core::Tensor::cat({target.colors.cpu(), suffix.colors.cpu()}, 0);
        }

        struct DomeProjectionCleanupResult {
            lfs::core::PointCloud regular_point_cloud;
            lfs::core::PointCloud projected_sky_point_cloud;
            size_t projected_sky_points = 0;
            size_t masked_sky_points = 0;
            size_t preserved_masked_non_sky_points = 0;
            size_t clamped_regular_points = 0;
            size_t dropped_invalid_points = 0;
        };

        struct SkyMaskFaceLookup {
            std::string id;
            std::vector<uint8_t> mask;
        };

        struct SkyMaskLookup {
            int face_size = 0;
            int marked_pixels = 0;
            std::vector<SkyMaskFaceLookup> faces;
        };

        uint8_t colorToByte(const float value) {
            return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
        }

        bool colorHasInformation(const glm::vec3& color) {
            if (!std::isfinite(color.r) || !std::isfinite(color.g) || !std::isfinite(color.b)) {
                return false;
            }
            const float r = std::clamp(color.r, 0.0f, 1.0f);
            const float g = std::clamp(color.g, 0.0f, 1.0f);
            const float b = std::clamp(color.b, 0.0f, 1.0f);
            const float max_channel = std::max(r, std::max(g, b));
            const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            return luma > 0.08f && max_channel > 0.10f;
        }

        glm::vec3 pointColorRgb(const uint8_t* colors_ptr, const size_t index) {
            return glm::vec3(
                static_cast<float>(colors_ptr[index * 3u + 0u]) / 255.0f,
                static_cast<float>(colors_ptr[index * 3u + 1u]) / 255.0f,
                static_cast<float>(colors_ptr[index * 3u + 2u]) / 255.0f);
        }

        bool pointColorMatchesDominantSky(const glm::vec3& color,
                                          const std::optional<glm::vec3>& dominant_color,
                                          const float dominant_color_radius) {
            if (!dominant_color || dominant_color_radius <= 0.0f || !colorHasInformation(color)) {
                return false;
            }
            const glm::vec3 dominant = glm::clamp(*dominant_color, glm::vec3(0.0f), glm::vec3(1.0f));
            const glm::vec3 clamped = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
            const float luma = 0.2126f * clamped.r + 0.7152f * clamped.g + 0.0722f * clamped.b;
            const float dominant_luma = 0.2126f * dominant.r + 0.7152f * dominant.g + 0.0722f * dominant.b;
            const float distance = glm::length(glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f)) -
                                               dominant);
            const float tolerance = std::clamp(dominant_color_radius, 0.10f, 0.45f);
            return distance <= tolerance && std::abs(luma - dominant_luma) <= std::max(0.18f, tolerance);
        }

        std::filesystem::path resolveSkyMaskManifestPath(const std::filesystem::path& manifest_path,
                                                         const nlohmann::json& value) {
            if (!value.is_string()) {
                return {};
            }
            std::filesystem::path path = lfs::core::utf8_to_path(value.get<std::string>());
            if (path.is_relative()) {
                path = manifest_path.parent_path() / path;
            }
            return path;
        }

        std::vector<uint8_t> loadSkyMaskPixels(const std::filesystem::path& mask_path,
                                               const int face_size) {
            const size_t pixel_count = static_cast<size_t>(face_size) * static_cast<size_t>(face_size);
            std::vector<uint8_t> mask(pixel_count, 0);
            if (mask_path.empty() || !std::filesystem::exists(mask_path)) {
                return mask;
            }

            unsigned char* data = nullptr;
            int width = 0;
            int height = 0;
            int channels = 0;
            try {
                std::tie(data, width, height, channels) = lfs::core::load_image(mask_path, -1, 0);
            } catch (const std::exception& e) {
                LOG_WARN("Sky mask lookup skipped '{}': {}", lfs::core::path_to_utf8(mask_path), e.what());
                return mask;
            }

            if (!data || width != face_size || height != face_size || channels < 1) {
                if (data) {
                    lfs::core::free_image(data);
                }
                return mask;
            }

            for (size_t idx = 0; idx < pixel_count; ++idx) {
                mask[idx] = data[idx * static_cast<size_t>(channels)] >= 128 ? 255 : 0;
            }
            lfs::core::free_image(data);
            return mask;
        }

        SkyMaskLookup loadSkyMaskLookup(const std::filesystem::path& manifest_path) {
            SkyMaskLookup lookup;
            if (manifest_path.empty() || !std::filesystem::exists(manifest_path)) {
                return lookup;
            }

            nlohmann::json manifest;
            try {
                std::ifstream file(manifest_path);
                if (!file) {
                    return lookup;
                }
                file >> manifest;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to parse sky mask lookup '{}': {}",
                         lfs::core::path_to_utf8(manifest_path),
                         e.what());
                return lookup;
            }

            lookup.face_size = manifest.value("face_size", 0);
            if (lookup.face_size < 16 || !manifest.contains("faces") || !manifest["faces"].is_object()) {
                lookup.face_size = 0;
                return lookup;
            }

            constexpr std::array<std::string_view, 6> kFaces{
                "pos_x", "neg_x", "neg_y", "pos_y", "pos_z", "neg_z"};
            lookup.faces.reserve(kFaces.size());
            for (const std::string_view face_id : kFaces) {
                const std::string id(face_id);
                if (!manifest["faces"].contains(id) || !manifest["faces"][id].is_object()) {
                    continue;
                }
                const auto& face_json = manifest["faces"][id];
                const std::filesystem::path mask_path =
                    resolveSkyMaskManifestPath(manifest_path,
                                               face_json.contains("mask") ? face_json["mask"] : nlohmann::json{});
                auto mask = loadSkyMaskPixels(mask_path, lookup.face_size);
                lookup.marked_pixels += static_cast<int>(std::count_if(mask.begin(), mask.end(), [](const uint8_t v) {
                    return v >= 128;
                }));
                lookup.faces.push_back(SkyMaskFaceLookup{
                    .id = id,
                    .mask = std::move(mask),
                });
            }
            return lookup;
        }

        const SkyMaskFaceLookup* findSkyMaskFace(const SkyMaskLookup& lookup, const std::string_view id) {
            for (const auto& face : lookup.faces) {
                if (face.id == id) {
                    return &face;
                }
            }
            return nullptr;
        }

        bool directionToSkyMaskPixel(const glm::vec3& dir,
                                     std::string_view& face_id,
                                     int& x,
                                     int& y,
                                     const int face_size) {
            const glm::vec3 abs_dir(std::abs(dir.x), std::abs(dir.y), std::abs(dir.z));
            float a = 0.0f;
            float b = 0.0f;
            float major = 0.0f;
            if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
                major = std::max(abs_dir.x, 1.0e-6f);
                if (dir.x >= 0.0f) {
                    face_id = "pos_x";
                    a = -dir.z / major;
                } else {
                    face_id = "neg_x";
                    a = dir.z / major;
                }
                b = -dir.y / major;
            } else if (abs_dir.y >= abs_dir.z) {
                major = std::max(abs_dir.y, 1.0e-6f);
                if (dir.y >= 0.0f) {
                    face_id = "pos_y";
                    a = dir.x / major;
                    b = -dir.z / major;
                } else {
                    face_id = "neg_y";
                    a = dir.x / major;
                    b = dir.z / major;
                }
            } else {
                major = std::max(abs_dir.z, 1.0e-6f);
                if (dir.z >= 0.0f) {
                    face_id = "pos_z";
                    a = dir.x / major;
                } else {
                    face_id = "neg_z";
                    a = -dir.x / major;
                }
                b = -dir.y / major;
            }

            if (!std::isfinite(a) || !std::isfinite(b) || face_size <= 0) {
                return false;
            }
            x = std::clamp(static_cast<int>(std::floor((std::clamp(a, -1.0f, 1.0f) + 1.0f) *
                                                       0.5f * static_cast<float>(face_size))),
                           0,
                           face_size - 1);
            y = std::clamp(static_cast<int>(std::floor((1.0f - std::clamp(b, -1.0f, 1.0f)) *
                                                       0.5f * static_cast<float>(face_size))),
                           0,
                           face_size - 1);
            return true;
        }

        bool directionInSkyMask(const SkyMaskLookup& lookup, const glm::vec3& dir) {
            if (lookup.face_size <= 0 || lookup.marked_pixels <= 0 || lookup.faces.empty()) {
                return false;
            }
            std::string_view face_id;
            int x = 0;
            int y = 0;
            if (!directionToSkyMaskPixel(dir, face_id, x, y, lookup.face_size)) {
                return false;
            }

            const auto* face = findSkyMaskFace(lookup, face_id);
            if (!face || face->mask.empty()) {
                return false;
            }

            constexpr int kAbsorbRadius = 3;
            for (int dy = -kAbsorbRadius; dy <= kAbsorbRadius; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= lookup.face_size) {
                    continue;
                }
                for (int dx = -kAbsorbRadius; dx <= kAbsorbRadius; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= lookup.face_size) {
                        continue;
                    }
                    const size_t idx = static_cast<size_t>(yy) * static_cast<size_t>(lookup.face_size) +
                                       static_cast<size_t>(xx);
                    if (idx < face->mask.size() && face->mask[idx] >= 128) {
                        return true;
                    }
                }
            }
            return false;
        }

        std::array<uint8_t, 3> dominantSkyColorBytes(const std::optional<glm::vec3>& dominant_color) {
            const glm::vec3 color = dominant_color.value_or(glm::vec3(0.82f, 0.88f, 0.96f));
            return {
                colorToByte(color.r),
                colorToByte(color.g),
                colorToByte(color.b),
            };
        }

        lfs::core::Tensor pointCloudColorsAsUint8(const lfs::core::PointCloud& point_cloud) {
            auto colors = point_cloud.colors.cpu();
            if (colors.dtype() == lfs::core::DataType::UInt8) {
                return colors.contiguous();
            }
            return colors.to(lfs::core::DataType::Float32)
                .mul(255.0f)
                .clamp(0.0f, 255.0f)
                .to(lfs::core::DataType::UInt8)
                .contiguous();
        }

        lfs::core::PointCloud makePointCloudFromVectors(std::vector<float>& means,
                                                        std::vector<uint8_t>& colors) {
            const size_t count = means.size() / 3u;
            if (count == 0) {
                return {};
            }

            auto means_tensor = lfs::core::Tensor::from_vector(means, {count, size_t{3}}, lfs::core::Device::CPU);
            auto colors_tensor = lfs::core::Tensor::empty({count, size_t{3}},
                                                          lfs::core::Device::CPU,
                                                          lfs::core::DataType::UInt8);
            std::memcpy(colors_tensor.data_ptr(), colors.data(), colors.size() * sizeof(uint8_t));
            return lfs::core::PointCloud(std::move(means_tensor), std::move(colors_tensor));
        }

        DomeProjectionCleanupResult projectInitialPointCloudToDome(
            const lfs::core::PointCloud& point_cloud,
            const glm::mat4& model_world,
            const glm::mat4& dome_world,
            const SkyMaskLookup& sky_mask_lookup,
            const std::optional<glm::vec3>& dominant_sky_color,
            const float dominant_sky_color_radius) {
            DomeProjectionCleanupResult result;
            const size_t count = static_cast<size_t>(point_cloud.size());
            if (count == 0) {
                return result;
            }

            auto means_cpu = point_cloud.means.cpu().contiguous();
            auto colors_cpu = pointCloudColorsAsUint8(point_cloud);
            const float* means_ptr = means_cpu.ptr<float>();
            const uint8_t* colors_ptr = colors_cpu.ptr<uint8_t>();

            std::vector<float> regular_means;
            std::vector<uint8_t> regular_colors;
            std::vector<float> sky_means;
            std::vector<uint8_t> sky_colors;
            regular_means.reserve(count * 3u);
            regular_colors.reserve(count * 3u);
            sky_means.reserve(std::min<size_t>(count, 4096u) * 3u);
            sky_colors.reserve(std::min<size_t>(count, 4096u) * 3u);

            const glm::mat4 dome_from_world = glm::inverse(dome_world);
            const glm::mat4 model_from_world = glm::inverse(model_world);
            const auto sky_color = dominantSkyColorBytes(dominant_sky_color);

            const auto push_regular = [&](const glm::vec3& p, const size_t i) {
                regular_means.insert(regular_means.end(), {p.x, p.y, p.z});
                regular_colors.insert(regular_colors.end(), {
                    colors_ptr[i * 3u + 0u],
                    colors_ptr[i * 3u + 1u],
                    colors_ptr[i * 3u + 2u],
                });
            };
            const auto push_sky = [&](const glm::vec3& p) {
                sky_means.insert(sky_means.end(), {p.x, p.y, p.z});
                sky_colors.insert(sky_colors.end(), {sky_color[0], sky_color[1], sky_color[2]});
            };

            for (size_t i = 0; i < count; ++i) {
                const glm::vec3 model_point(
                    means_ptr[i * 3u + 0u],
                    means_ptr[i * 3u + 1u],
                    means_ptr[i * 3u + 2u]);
                if (!std::isfinite(model_point.x) ||
                    !std::isfinite(model_point.y) ||
                    !std::isfinite(model_point.z)) {
                    ++result.dropped_invalid_points;
                    continue;
                }

                const glm::vec3 world_point = glm::vec3(model_world * glm::vec4(model_point, 1.0f));
                glm::vec3 dome_point = glm::vec3(dome_from_world * glm::vec4(world_point, 1.0f));
                const float radius = glm::length(dome_point);
                if (!std::isfinite(radius) || radius <= 1.0e-6f) {
                    ++result.dropped_invalid_points;
                    continue;
                }

                const glm::vec3 dir = dome_point / radius;
                const bool marked_sky_direction = directionInSkyMask(sky_mask_lookup, dir);
                const glm::vec3 point_color = pointColorRgb(colors_ptr, i);
                const bool sky_colored_point = pointColorMatchesDominantSky(
                    point_color,
                    dominant_sky_color,
                    dominant_sky_color_radius);
                const bool outside_dome = radius > 1.001f;
                if (!marked_sky_direction && !outside_dome) {
                    push_regular(model_point, i);
                    continue;
                }
                if (marked_sky_direction && !outside_dome && !sky_colored_point) {
                    push_regular(model_point, i);
                    ++result.preserved_masked_non_sky_points;
                    continue;
                }

                const bool sky_half_space = dir.y <= 0.0f;
                const bool project_to_sky_shell =
                    (marked_sky_direction && sky_colored_point) ||
                    (outside_dome && sky_half_space && sky_colored_point);
                const glm::vec3 projected_dome = dir * (project_to_sky_shell ? 1.0f : 0.995f);
                const glm::vec3 projected_world = glm::vec3(dome_world * glm::vec4(projected_dome, 1.0f));
                const glm::vec3 projected_model = glm::vec3(model_from_world * glm::vec4(projected_world, 1.0f));
                if (!std::isfinite(projected_model.x) ||
                    !std::isfinite(projected_model.y) ||
                    !std::isfinite(projected_model.z)) {
                    ++result.dropped_invalid_points;
                    continue;
                }

                if (project_to_sky_shell) {
                    push_sky(projected_model);
                    ++result.projected_sky_points;
                    if (marked_sky_direction) {
                        ++result.masked_sky_points;
                    }
                } else {
                    push_regular(projected_model, i);
                    ++result.clamped_regular_points;
                }
            }

            result.regular_point_cloud = makePointCloudFromVectors(regular_means, regular_colors);
            result.projected_sky_point_cloud = makePointCloudFromVectors(sky_means, sky_colors);
            return result;
        }

        int skyGaussianCap(const int max_cap, const size_t regular_count) {
            constexpr int kDefaultSkyCap = 250'000;
            if (max_cap <= 0 || regular_count == 0) {
                return kDefaultSkyCap;
            }
            return std::max(1, std::min(kDefaultSkyCap, max_cap / 10));
        }

        float pointCloudDiagonal(const lfs::core::PointCloud& point_cloud,
                                 const bool percentile_bounds = true) {
            glm::vec3 min_bounds{0.0f};
            glm::vec3 max_bounds{0.0f};
            if (!lfs::core::compute_bounds(point_cloud, min_bounds, max_bounds, 0.0f, percentile_bounds)) {
                return 0.0f;
            }
            const float diagonal = glm::length(max_bounds - min_bounds);
            return std::isfinite(diagonal) ? diagonal : 0.0f;
        }

        float skyInitialScale(const lfs::core::PointCloud& regular_point_cloud,
                              const lfs::core::PointCloud& sky_point_cloud) {
            constexpr float kMinScale = 1.0e-4f;
            float scale = kMinScale;

            const float regular_diag = pointCloudDiagonal(regular_point_cloud);
            if (regular_diag > 0.0f) {
                scale = std::max(scale, regular_diag * 0.03f);
            }

            const float sky_diag = pointCloudDiagonal(sky_point_cloud, false);
            if (sky_diag > 0.0f) {
                // Keep the initial dome visible without letting sparse sky splats cover the scene.
                scale = std::max(scale, sky_diag * 0.0005f);
            }

            return std::max(kMinScale, scale);
        }

        void initializeSkyPrefixAppearance(lfs::core::SplatData& splat_data,
                                           const size_t frozen_sky_prefix,
                                           const float initial_scale) {
            const size_t count = std::min(frozen_sky_prefix, static_cast<size_t>(splat_data.size()));
            if (count == 0) {
                return;
            }

            constexpr float kInitialOpacity = 0.06f;
            const float safe_scale = std::max(initial_scale, 1.0e-4f);
            const float log_scale = std::log(safe_scale);
            const float raw_opacity = std::log(kInitialOpacity / (1.0f - kInitialOpacity));

            splat_data.scaling_raw().slice(0, 0, count).fill_(log_scale);
            splat_data.opacity_raw().slice(0, 0, count).fill_(raw_opacity);

            LOG_INFO("Initialized sky gaussian appearance: {} fixed positions, scale {:.4f}, opacity {:.3f}",
                     count, safe_scale, kInitialOpacity);
        }
    } // namespace

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {

        auto data_loader = lfs::io::Loader::create();

        const auto& data_path = params.dataset.data_path;
        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .centralize = parse_centralize(params.dataset.centralize_dataset),
            .progress = [&data_path](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = data_path,
                    .progress = percentage,
                    .step = message}
                    .emit();
            }};

        LOG_INFO("Loading dataset from: {}", lfs::core::path_to_utf8(params.dataset.data_path));
        auto load_result = data_loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error().format()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                applyTrainingSHDegree(*model, params.optimization.sh_degree);
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                LOG_INFO("Loaded PLY directly into scene");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result->scene_center);
                scene.setImagesHaveAlpha(load_result->images_have_alpha);

                // Build dataset hierarchy in scene graph
                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (params.init_path.has_value()) {
                    const std::filesystem::path init_file = lfs::core::utf8_to_path(params.init_path.value());
                    const auto ext = init_file.extension().string();

                    if (ext == ".ply" && !lfs::io::is_gaussian_splat_ply(init_file)) {
                        auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                        if (!pc_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), pc_result.error()));
                        }

                        auto splat_result = lfs::core::init_model_from_pointcloud(
                            params, load_result->scene_center, *pc_result, static_cast<int>(pc_result->size()));

                        if (!splat_result) {
                            return std::unexpected(std::format("Init failed: {}", splat_result.error()));
                        }

                        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                        LOG_INFO("Initialized {} Gaussians from {} (sh={})",
                                 model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                        scene.addSplat("Model", std::move(model), dataset_id);
                        scene.setTrainingModelNode("Model");
                    } else {
                        auto loader = lfs::io::Loader::create();
                        auto load_result = loader->load(init_file);

                        if (!load_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), load_result.error().format()));
                        }

                        try {
                            auto splat_data = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(load_result->data));
                            auto model = std::make_unique<lfs::core::SplatData>(std::move(splat_data));

                            applyTrainingSHDegree(*model, params.optimization.sh_degree);

                            LOG_INFO("Loaded {} Gaussians from {} (sh={})",
                                     model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                            scene.addSplat("Model", std::move(model), dataset_id);
                            scene.setTrainingModelNode("Model");
                        } catch (const std::bad_variant_access&) {
                            return std::unexpected(std::format("'{}': invalid SplatData", lfs::core::path_to_utf8(init_file)));
                        }
                    }
                } else {
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        LOG_INFO("Adding {} points to scene", data.point_cloud->size());
                        scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                    } else {
                        LOG_INFO("No point cloud, using random initialization");
                        auto pc = createRandomPointCloud();
                        LOG_INFO("Adding {} random points to scene", pc->size());
                        scene.addPointCloud("PointCloud", pc, dataset_id);
                    }
                }

                const auto& cameras = data.cameras;
                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;

                size_t train_count = 0;
                size_t val_count = 0;
                size_t mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_eval = enable_eval && (i % test_every) == 0;
                    cameras[i]->set_split(is_eval ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                    if (is_eval) {
                        val_count++;
                    } else {
                        train_count++;
                    }
                    if (cameras[i]->has_mask()) {
                        mask_count++;
                    }
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);

                const auto train_cameras_id = scene.addCameraGroup(
                    std::format("Training ({})", train_count),
                    cameras_group_id,
                    train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!enable_eval || (i % test_every) != 0) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        std::format("Validation ({})", val_count),
                        cameras_group_id,
                        val_count);

                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if ((i % test_every) == 0) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                LOG_INFO("Loaded dataset '{}' into scene: {} train{} cameras{}",
                         dataset_name, train_count,
                         enable_eval ? std::format(" + {} val", val_count) : "",
                         mask_count > 0 ? std::format(" ({} with masks)", mask_count) : "");
                return {};

            } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                assert(data && "MeshData must not be null");
                std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                if (mesh_name.empty())
                    mesh_name = "mesh";
                scene.addMesh(mesh_name, data);
                LOG_INFO("Loaded mesh '{}' into scene", mesh_name);
                return {};

            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }

    std::expected<void, std::string> initializeTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {

        if (auto* model = scene.getTrainingModel()) {
            applyTrainingSHDegree(*model, params.optimization.sh_degree);
            scene.notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
            return {};
        }

        lfs::core::NodeId point_cloud_node_id = lfs::core::NULL_NODE;
        lfs::core::NodeId parent_id = lfs::core::NULL_NODE;
        const lfs::core::PointCloud* point_cloud = nullptr;
        glm::mat4 node_transform{1.0f};

        for (const auto* node : scene.getNodes()) {
            if (node->type == lfs::core::NodeType::POINTCLOUD &&
                node->point_cloud &&
                node->name != lfs::core::kProjectionDomeSkyPreviewNodeName) {
                point_cloud_node_id = node->id;
                parent_id = node->parent_id;
                node_transform = node->transform();
                point_cloud = node->point_cloud.get();
                break;
            }
        }

        lfs::core::PointCloud point_cloud_to_use;
        const int max_cap = params.optimization.max_cap;

        if (point_cloud && point_cloud->size() > 0) {
            const lfs::core::CropBoxData* cropbox_data = nullptr;
            lfs::core::NodeId cropbox_id = lfs::core::NULL_NODE;

            if (point_cloud_node_id != lfs::core::NULL_NODE) {
                cropbox_id = scene.getCropBoxForSplat(point_cloud_node_id);
                if (cropbox_id != lfs::core::NULL_NODE) {
                    cropbox_data = scene.getCropBoxData(cropbox_id);
                }
            }

            if (cropbox_data && cropbox_data->enabled) {
                const glm::mat4 world_to_cropbox = glm::inverse(scene.getWorldTransform(cropbox_id));
                const auto& means = point_cloud->means;
                const auto& colors = point_cloud->colors;
                const size_t num_points = point_cloud->size();

                auto means_cpu = means.cpu();
                auto colors_cpu = colors.cpu();
                const float* means_ptr = means_cpu.ptr<float>();
                const uint8_t* colors_ptr = colors_cpu.ptr<uint8_t>();

                std::vector<float> filtered_means;
                std::vector<uint8_t> filtered_colors;
                filtered_means.reserve(num_points * 3);
                filtered_colors.reserve(num_points * 3);

                for (size_t i = 0; i < num_points; ++i) {
                    const glm::vec3 pos(means_ptr[i * 3], means_ptr[i * 3 + 1], means_ptr[i * 3 + 2]);
                    const glm::vec4 local_pos = world_to_cropbox * glm::vec4(pos, 1.0f);
                    const glm::vec3 local = glm::vec3(local_pos) / local_pos.w;

                    bool inside = local.x >= cropbox_data->min.x && local.x <= cropbox_data->max.x &&
                                  local.y >= cropbox_data->min.y && local.y <= cropbox_data->max.y &&
                                  local.z >= cropbox_data->min.z && local.z <= cropbox_data->max.z;

                    if (cropbox_data->inverse)
                        inside = !inside;

                    if (inside) {
                        filtered_means.push_back(means_ptr[i * 3]);
                        filtered_means.push_back(means_ptr[i * 3 + 1]);
                        filtered_means.push_back(means_ptr[i * 3 + 2]);
                        filtered_colors.push_back(colors_ptr[i * 3]);
                        filtered_colors.push_back(colors_ptr[i * 3 + 1]);
                        filtered_colors.push_back(colors_ptr[i * 3 + 2]);
                    }
                }

                const size_t filtered_count = filtered_means.size() / 3;
                LOG_INFO("CropBox filtering: {} -> {} points", num_points, filtered_count);

                if (filtered_count == 0) {
                    return std::unexpected("CropBox filtered out all points");
                }

                auto filtered_means_tensor = lfs::core::Tensor::from_vector(
                    filtered_means, {filtered_count, 3}, lfs::core::Device::CPU);
                auto filtered_colors_tensor = lfs::core::Tensor::zeros(
                    {filtered_count, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                std::memcpy(filtered_colors_tensor.data_ptr(), filtered_colors.data(),
                            filtered_colors.size() * sizeof(uint8_t));

                point_cloud_to_use = lfs::core::PointCloud(filtered_means_tensor, filtered_colors_tensor);
            } else {
                point_cloud_to_use = *point_cloud;
                if (max_cap > 0) {
                    point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                    point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
                }
            }
        } else {
            LOG_INFO("No point cloud provided, using random initialization");
            point_cloud_to_use = *createRandomPointCloud();
        }

        size_t frozen_sky_prefix = 0;
        float frozen_sky_initial_scale = 0.0f;
        if (!params.optimization.sky_mask_path.empty()) {
            const size_t regular_point_count = static_cast<size_t>(point_cloud_to_use.size());
            const int sky_cap = skyGaussianCap(max_cap, regular_point_count);
            const glm::mat4 parent_world =
                parent_id != lfs::core::NULL_NODE ? scene.getWorldTransform(parent_id) : glm::mat4(1.0f);
            const glm::mat4 model_world = parent_world * node_transform;
            const glm::mat4 output_from_world = glm::inverse(model_world);

            auto sky_result = lfs::core::createProjectionDomeSkyPointCloud(
                scene,
                lfs::core::ProjectionDomeSkyPointCloudOptions{
                    .manifest_path = params.optimization.sky_mask_path,
                    .output_from_world = output_from_world,
                    .max_gaussians = sky_cap,
                });
            if (!sky_result) {
                return std::unexpected(std::format("Failed to initialize sky sphere from mask: {}",
                                                   sky_result.error()));
            }

            if (sky_result->gaussian_count > 0) {
                auto sky_point_cloud = std::move(sky_result->point_cloud);
                if (const auto dome_id = lfs::core::findProjectionDome(scene)) {
                    const glm::mat4 dome_world = scene.getWorldTransform(*dome_id);
                    const SkyMaskLookup sky_mask_lookup = loadSkyMaskLookup(params.optimization.sky_mask_path);
                    auto cleanup = projectInitialPointCloudToDome(
                        point_cloud_to_use,
                        model_world,
                        dome_world,
                        sky_mask_lookup,
                        sky_result->dominant_color,
                        sky_result->dominant_color_radius);
                    if (cleanup.projected_sky_points > 0 || cleanup.preserved_masked_non_sky_points > 0 ||
                        cleanup.clamped_regular_points > 0 ||
                        cleanup.dropped_invalid_points > 0) {
                        point_cloud_to_use = std::move(cleanup.regular_point_cloud);
                        appendPointCloud(sky_point_cloud, cleanup.projected_sky_point_cloud);
                        LOG_INFO("Projection dome cleaned initial SfM cloud: {} sky/outlier points projected to frozen shell ({} by painted mask), {} masked non-sky-colored points preserved, {} regular points clamped inside sphere, {} invalid points dropped",
                                 cleanup.projected_sky_points,
                                 cleanup.masked_sky_points,
                                 cleanup.preserved_masked_non_sky_points,
                                 cleanup.clamped_regular_points,
                                 cleanup.dropped_invalid_points);
                    }
                }
                frozen_sky_prefix = static_cast<size_t>(sky_point_cloud.size());
                frozen_sky_initial_scale = skyInitialScale(point_cloud_to_use, sky_point_cloud);

                if (max_cap > 0) {
                    const size_t cap = static_cast<size_t>(max_cap);
                    if (frozen_sky_prefix >= cap) {
                        const size_t current_regular_count = static_cast<size_t>(point_cloud_to_use.size());
                        const size_t sky_keep =
                            current_regular_count > 0 ? std::max<size_t>(1, cap / 5) : cap;
                        sky_point_cloud = selectPointCloudRows(sky_point_cloud, std::min(frozen_sky_prefix, sky_keep));
                        frozen_sky_prefix = static_cast<size_t>(sky_point_cloud.size());
                        const size_t regular_cap = cap - frozen_sky_prefix;
                        if (regular_cap > 0 && point_cloud_to_use.size() > 0) {
                            point_cloud_to_use = selectPointCloudRows(point_cloud_to_use, regular_cap);
                            prependPointCloud(point_cloud_to_use, sky_point_cloud);
                        } else {
                            point_cloud_to_use = std::move(sky_point_cloud);
                        }
                    } else {
                        const size_t regular_cap = cap - frozen_sky_prefix;
                        if (static_cast<size_t>(point_cloud_to_use.size()) > regular_cap) {
                            LOG_WARN("Max cap ({}) leaves room for {} regular points after {} sky gaussians; downsampling regular point cloud",
                                     max_cap, regular_cap, frozen_sky_prefix);
                            point_cloud_to_use = selectPointCloudRows(point_cloud_to_use, regular_cap);
                        } else {
                            point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                            point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
                        }
                        prependPointCloud(point_cloud_to_use, sky_point_cloud);
                    }
                } else {
                    prependPointCloud(point_cloud_to_use, sky_point_cloud);
                }

                LOG_INFO("Initialized {} fixed-position sky gaussians from {} marked sky pixels",
                         frozen_sky_prefix, sky_result->marked_pixels);
            } else if (sky_result->marked_pixels > 0) {
                LOG_WARN("Sky mask has {} marked pixels, but no valid sky gaussians were generated",
                         sky_result->marked_pixels);
            } else {
                LOG_INFO("Sky mask manifest has no marked pixels; skipping sky sphere initialization");
            }
        }

        lfs::core::Tensor scene_center = scene.getSceneCenter();
        if (!scene_center.is_valid() || scene_center.numel() == 0) {
            LOG_WARN("No scene center from loader, computing from point cloud");
            if (point_cloud_to_use.size() > 0) {
                auto means_cpu = point_cloud_to_use.means.cpu();
                auto mean = means_cpu.mean({0});
                scene_center = max_cap > 0 ? mean : mean.cuda();
            } else {
                scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
            }
        } else {
            scene_center = max_cap > 0 ? scene_center.cpu() : scene_center.cuda();
        }

        auto splat_result = lfs::core::init_model_from_pointcloud(
            params, scene_center, point_cloud_to_use, max_cap);

        if (!splat_result) {
            return std::unexpected(std::format("Failed to initialize model: {}", splat_result.error()));
        }

        if (frozen_sky_prefix > 0) {
            splat_result->set_frozen_means_prefix(frozen_sky_prefix);
            initializeSkyPrefixAppearance(*splat_result, frozen_sky_prefix, frozen_sky_initial_scale);
        }

        if (max_cap > 0 && max_cap < static_cast<int>(splat_result->size())) {
            if (frozen_sky_prefix > 0) {
                return std::unexpected(std::format(
                    "Sky initialization produced {} gaussians, exceeding max cap {}",
                    splat_result->size(), max_cap));
            }
            LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                     max_cap, splat_result->size(), max_cap);
            lfs::core::random_choose(*splat_result, max_cap);
        }

        if (point_cloud_node_id != lfs::core::NULL_NODE) {
            if (const auto* pc_node = scene.getNodeById(point_cloud_node_id)) {
                scene.removeNode(pc_node->name, false);
            }
        }

        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
        LOG_INFO("Created training model with {} gaussians", model->size());
        scene.addSplat("Model", std::move(model), parent_id);
        if (node_transform != glm::mat4{1.0f}) {
            scene.setNodeTransform("Model", node_transform);
        }
        scene.setTrainingModelNode("Model");

        return {};
    }

    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params) {

        auto data_loader = lfs::io::Loader::create();

        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = true};

        auto result = data_loader->load(params.dataset.data_path, load_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::io::LoadResult&& load_result) {

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                applyTrainingSHDegree(*model, params.optimization.sh_degree);
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result.scene_center);
                scene.setImagesHaveAlpha(load_result.images_have_alpha);

                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (params.init_path.has_value()) {
                    const std::filesystem::path init_file = lfs::core::utf8_to_path(params.init_path.value());
                    const auto ext = init_file.extension().string();

                    if (ext == ".ply" && !lfs::io::is_gaussian_splat_ply(init_file)) {
                        auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                        if (!pc_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), pc_result.error()));
                        }

                        auto splat_result = lfs::core::init_model_from_pointcloud(
                            params, load_result.scene_center, *pc_result, static_cast<int>(pc_result->size()));
                        if (!splat_result) {
                            return std::unexpected(std::format("Init failed: {}", splat_result.error()));
                        }

                        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                        LOG_INFO("Init {} gaussians from {} (sh={})",
                                 model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                        scene.addSplat("Model", std::move(model), dataset_id);
                        scene.setTrainingModelNode("Model");
                    } else {
                        auto loader = lfs::io::Loader::create();
                        auto init_result = loader->load(init_file);
                        if (!init_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), init_result.error().format()));
                        }

                        try {
                            auto splat_data = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(init_result->data));
                            auto model = std::make_unique<lfs::core::SplatData>(std::move(splat_data));

                            applyTrainingSHDegree(*model, params.optimization.sh_degree);

                            LOG_INFO("Loaded {} gaussians from {} (sh={})",
                                     model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                            scene.addSplat("Model", std::move(model), dataset_id);
                            scene.setTrainingModelNode("Model");
                        } catch (const std::bad_variant_access&) {
                            return std::unexpected(std::format("'{}': invalid SplatData", lfs::core::path_to_utf8(init_file)));
                        }
                    }
                } else if (data.point_cloud && data.point_cloud->size() > 0) {
                    scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                } else {
                    scene.addPointCloud("PointCloud", createRandomPointCloud(), dataset_id);
                }

                const auto& cameras = data.cameras;
                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;

                size_t train_count = 0, val_count = 0, mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_val = enable_eval && (i % test_every) == 0;
                    cameras[i]->set_split(is_val ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                    is_val ? ++val_count : ++train_count;
                    if (cameras[i]->has_mask())
                        ++mask_count;
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);
                const auto train_cameras_id = scene.addCameraGroup(
                    std::format("Training ({})", train_count), cameras_group_id, train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!enable_eval || (i % test_every) != 0) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        std::format("Validation ({})", val_count), cameras_group_id, val_count);
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if ((i % test_every) == 0) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                LOG_INFO("Dataset '{}': {} train{} cameras{}",
                         dataset_name, train_count,
                         enable_eval ? std::format(" + {} val", val_count) : "",
                         mask_count > 0 ? std::format(" ({} masked)", mask_count) : "");
                return {};

            } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                assert(data && "MeshData must not be null");
                std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                if (mesh_name.empty())
                    mesh_name = "mesh";
                scene.addMesh(mesh_name, data);
                LOG_INFO("Loaded mesh '{}' into scene", mesh_name);
                return {};

            } else {
                return std::unexpected("Unknown data type from loader");
            }
        },
                          load_result.data);
    }

} // namespace lfs::training
