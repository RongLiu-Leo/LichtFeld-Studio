/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include <expected>
#include <filesystem>
#include <glm/mat4x4.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core {

    inline constexpr std::string_view kProjectionDomeNodeName = "Projection Dome";
    inline constexpr std::string_view kProjectionDomeSkyPreviewNodeName = "Sky Initialization";

    struct ProjectionDomeMeshOptions {
        int longitude_segments = 64;
        int latitude_segments = 32;
        int placeholder_texture_width = 512;
        int placeholder_texture_height = 256;
        bool double_sided = true;
    };

    struct ProjectionDomePlacement {
        glm::vec3 center{0.0f};
        float radius = 1.0f;
        NodeId parent_id = NULL_NODE;
    };

    struct ProjectionDomeBakeOptions {
        std::string node_name{std::string(kProjectionDomeNodeName)};
        int texture_width = 2048;
        int texture_height = 1024;
        int max_image_width = 1600;
        int resize_factor = -1;
        bool active_cameras_only = true;
        bool reject_masked_pixels = true;
        bool invert_masks = false;
        float mask_threshold = 0.5f;
        int hole_fill_iterations = 24;
        float edge_falloff = 0.08f;
    };

    struct ProjectionDomeBakeResult {
        int width = 0;
        int height = 0;
        int cameras_considered = 0;
        int cameras_used = 0;
        int texels_projected = 0;
        int texels_after_fill = 0;
        int masked_pixels = 0;
        int samples_rejected_by_mask = 0;
    };

    struct ProjectionDomeSkyCubemapOptions {
        std::string node_name{std::string(kProjectionDomeNodeName)};
        std::filesystem::path output_dir;
        int face_size = 512;
        bool overwrite_preview = true;
        bool reset_mask = false;
    };

    struct ProjectionDomeSkyCubemapFace {
        std::string id;
        std::string label;
        std::filesystem::path preview_path;
        std::filesystem::path mask_path;
        std::filesystem::path overlay_path;
        int valid_pixels = 0;
        int marked_pixels = 0;
    };

    struct ProjectionDomeSkyCubemapResult {
        std::filesystem::path output_dir;
        int face_size = 0;
        glm::mat4 dome_world{1.0f};
        std::vector<ProjectionDomeSkyCubemapFace> faces;
    };

    struct ProjectionDomeSkyMaskPaintResult {
        int marked_pixels = 0;
        int changed_pixels = 0;
    };

    struct ProjectionDomeSkyPointCloudOptions {
        std::string node_name{std::string(kProjectionDomeNodeName)};
        std::filesystem::path manifest_path;
        glm::mat4 output_from_world{1.0f};
        int max_gaussians = 50000;
    };

    struct ProjectionDomeSkyPointCloudResult {
        PointCloud point_cloud;
        int marked_pixels = 0;
        int gaussian_count = 0;
    };

    [[nodiscard]] LFS_CORE_API std::shared_ptr<MeshData>
    createProjectionDomeMesh(const ProjectionDomeMeshOptions& options = {});

    [[nodiscard]] LFS_CORE_API ProjectionDomePlacement
    estimateProjectionDomePlacement(const Scene& scene);

    [[nodiscard]] LFS_CORE_API std::optional<NodeId>
    findProjectionDome(const Scene& scene, std::string_view node_name = kProjectionDomeNodeName);

    [[nodiscard]] LFS_CORE_API std::expected<NodeId, std::string>
    ensureProjectionDome(Scene& scene,
                         const ProjectionDomeMeshOptions& mesh_options = {},
                         const ProjectionDomePlacement& placement = {});

    [[nodiscard]] LFS_CORE_API std::expected<ProjectionDomeBakeResult, std::string>
    bakeProjectionDomeTexture(Scene& scene, const ProjectionDomeBakeOptions& options = {});

    [[nodiscard]] LFS_CORE_API std::expected<ProjectionDomeSkyCubemapResult, std::string>
    prepareProjectionDomeSkyCubemap(Scene& scene,
                                    const ProjectionDomeSkyCubemapOptions& options = {});

    [[nodiscard]] LFS_CORE_API std::expected<ProjectionDomeSkyMaskPaintResult, std::string>
    paintProjectionDomeSkyMask(const std::filesystem::path& mask_path,
                               const std::filesystem::path& overlay_path,
                               int face_size,
                               int x,
                               int y,
                               int radius,
                               bool erase);

    [[nodiscard]] LFS_CORE_API std::expected<ProjectionDomeSkyMaskPaintResult, std::string>
    clearProjectionDomeSkyMask(const std::filesystem::path& mask_path,
                               const std::filesystem::path& overlay_path,
                               int face_size);

    [[nodiscard]] LFS_CORE_API std::expected<ProjectionDomeSkyPointCloudResult, std::string>
    createProjectionDomeSkyPointCloud(const Scene& scene,
                                      const ProjectionDomeSkyPointCloudOptions& options);

} // namespace lfs::core
