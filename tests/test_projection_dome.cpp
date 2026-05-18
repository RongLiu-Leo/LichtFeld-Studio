/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/projection_dome.hpp"
#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>

using namespace lfs::core;

TEST(ProjectionDomeTest, CreateMeshHasTextureUvAndMaterial) {
    ProjectionDomeMeshOptions options;
    options.longitude_segments = 8;
    options.latitude_segments = 4;
    options.placeholder_texture_width = 32;
    options.placeholder_texture_height = 16;
    options.double_sided = true;

    const auto mesh = createProjectionDomeMesh(options);
    ASSERT_TRUE(mesh);

    EXPECT_EQ(mesh->vertex_count(), (8 + 1) * (4 + 1));
    EXPECT_EQ(mesh->face_count(), 8 * 4 * 4);
    ASSERT_TRUE(mesh->has_normals());
    ASSERT_TRUE(mesh->has_texcoords());
    ASSERT_FALSE(mesh->materials.empty());
    ASSERT_FALSE(mesh->submeshes.empty());
    ASSERT_FALSE(mesh->texture_images.empty());

    EXPECT_EQ(mesh->materials.front().albedo_tex, 1u);
    EXPECT_TRUE(mesh->materials.front().double_sided);
    EXPECT_LT(mesh->materials.front().base_color.a, 1.0f);
    EXPECT_EQ(mesh->submeshes.front().index_count, static_cast<size_t>(mesh->face_count()) * 3u);
    EXPECT_EQ(mesh->texture_images.front().width, 32);
    EXPECT_EQ(mesh->texture_images.front().height, 16);
    EXPECT_EQ(mesh->texture_images.front().channels, 3);

    auto vertices = mesh->vertices.cpu();
    auto normals = mesh->normals.cpu();
    auto texcoords = mesh->texcoords.cpu();
    auto v = vertices.accessor<float, 2>();
    auto n = normals.accessor<float, 2>();
    auto uv = texcoords.accessor<float, 2>();

    EXPECT_NEAR(v(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(v(0, 1), 0.0f, 1e-5f);
    EXPECT_NEAR(v(0, 2), 0.0f, 1e-5f);
    EXPECT_NEAR(n(0, 0), -1.0f, 1e-5f);
    EXPECT_NEAR(n(0, 1), 0.0f, 1e-5f);
    EXPECT_NEAR(n(0, 2), 0.0f, 1e-5f);
    EXPECT_NEAR(uv(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(uv(0, 1), 1.0f, 1e-5f);
}

TEST(ProjectionDomeTest, EnsureAddsTransformableMeshNode) {
    Scene scene;
    const NodeId dataset_id = scene.addDataset("Dataset");

    ProjectionDomePlacement placement;
    placement.parent_id = dataset_id;
    placement.center = glm::vec3(1.0f, 2.0f, 3.0f);
    placement.radius = 5.0f;

    const auto result = ensureProjectionDome(scene, {}, placement);
    ASSERT_TRUE(result.has_value()) << result.error();

    const auto* node = scene.getNodeById(*result);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->type, NodeType::MESH);
    ASSERT_TRUE(node->mesh);
    EXPECT_EQ(node->name, std::string(kProjectionDomeNodeName));
    EXPECT_EQ(node->parent_id, dataset_id);

    const glm::mat4 transform = node->local_transform.get();
    EXPECT_NEAR(transform[3].x, 1.0f, 1e-5f);
    EXPECT_NEAR(transform[3].y, 2.0f, 1e-5f);
    EXPECT_NEAR(transform[3].z, 3.0f, 1e-5f);
    EXPECT_NEAR(glm::length(glm::vec3(transform[0])), 5.0f, 1e-5f);
    EXPECT_NEAR(glm::length(glm::vec3(transform[1])), 5.0f, 1e-5f);
    EXPECT_NEAR(glm::length(glm::vec3(transform[2])), 5.0f, 1e-5f);

    const auto second = ensureProjectionDome(scene, {}, placement);
    ASSERT_TRUE(second.has_value()) << second.error();
    EXPECT_EQ(*second, *result);
}

TEST(ProjectionDomeTest, BakeFailsClearlyWithoutUsableImages) {
    Scene scene;
    ASSERT_TRUE(ensureProjectionDome(scene).has_value());

    auto result = bakeProjectionDomeTexture(scene, ProjectionDomeBakeOptions{
                                                      .texture_width = 32,
                                                      .texture_height = 16,
                                                  });
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("No usable camera images"), std::string::npos);
}

TEST(ProjectionDomeTest, PrepareSkyCubemapAndPaintMask) {
    Scene scene;
    ASSERT_TRUE(ensureProjectionDome(scene, ProjectionDomeMeshOptions{
                                                .longitude_segments = 8,
                                                .latitude_segments = 4,
                                                .placeholder_texture_width = 32,
                                                .placeholder_texture_height = 16,
                                            })
                    .has_value());

    const auto output_dir =
        std::filesystem::temp_directory_path() /
        ("lfs_projection_dome_sky_test_" +
         std::to_string(reinterpret_cast<std::uintptr_t>(&scene)));
    std::filesystem::remove_all(output_dir);

    auto result = prepareProjectionDomeSkyCubemap(scene, ProjectionDomeSkyCubemapOptions{
                                                            .output_dir = output_dir,
                                                            .face_size = 32,
                                                            .overwrite_preview = true,
                                                            .reset_mask = true,
                                                        });
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->face_size, 32);
    ASSERT_EQ(result->faces.size(), 6u);

    const auto front = std::find_if(result->faces.begin(), result->faces.end(), [](const auto& face) {
        return face.id == "pos_z";
    });
    ASSERT_NE(front, result->faces.end());
    EXPECT_TRUE(std::filesystem::exists(front->preview_path));
    EXPECT_TRUE(std::filesystem::exists(front->mask_path));
    EXPECT_TRUE(std::filesystem::exists(front->overlay_path));
    EXPECT_GT(front->valid_pixels, 0);
    EXPECT_EQ(front->marked_pixels, 0);

    const auto paint = paintProjectionDomeSkyMask(front->mask_path, front->overlay_path, 32, 16, 8, 4, false);
    ASSERT_TRUE(paint.has_value()) << paint.error();
    EXPECT_GT(paint->changed_pixels, 0);
    EXPECT_GT(paint->marked_pixels, 0);

    const auto clear = clearProjectionDomeSkyMask(front->mask_path, front->overlay_path, 32);
    ASSERT_TRUE(clear.has_value()) << clear.error();
    EXPECT_EQ(clear->marked_pixels, 0);

    std::filesystem::remove_all(output_dir);
}

TEST(ProjectionDomeTest, SkyMaskCreatesPointCloudOnDome) {
    Scene scene;
    ASSERT_TRUE(ensureProjectionDome(scene, {}, ProjectionDomePlacement{
                                                   .center = glm::vec3(1.0f, 2.0f, 3.0f),
                                                   .radius = 10.0f,
                                               })
                    .has_value());

    const auto output_dir =
        std::filesystem::temp_directory_path() /
        ("lfs_projection_dome_sky_pc_test_" +
         std::to_string(reinterpret_cast<std::uintptr_t>(&scene)));
    std::filesystem::remove_all(output_dir);

    auto result = prepareProjectionDomeSkyCubemap(scene, ProjectionDomeSkyCubemapOptions{
                                                            .output_dir = output_dir,
                                                            .face_size = 32,
                                                            .overwrite_preview = true,
                                                            .reset_mask = true,
                                                        });
    ASSERT_TRUE(result.has_value()) << result.error();

    const auto top = std::find_if(result->faces.begin(), result->faces.end(), [](const auto& face) {
        return face.id == "neg_y";
    });
    ASSERT_NE(top, result->faces.end());
    ASSERT_TRUE(paintProjectionDomeSkyMask(top->mask_path, top->overlay_path, 32, 16, 16, 6, false).has_value());

    nlohmann::json manifest;
    manifest["type"] = "projection_dome_sky_cubemap_mask";
    manifest["face_size"] = result->face_size;
    manifest["faces"] = nlohmann::json::object();
    for (const auto& face : result->faces) {
        manifest["faces"][face.id] = {
            {"label", face.label},
            {"mask", face.mask_path.string()},
            {"preview", face.preview_path.string()},
        };
    }
    const auto manifest_path = output_dir / "sky_mask_manifest.json";
    {
        std::ofstream file(manifest_path);
        file << manifest.dump(2);
    }

    auto sky = createProjectionDomeSkyPointCloud(scene, ProjectionDomeSkyPointCloudOptions{
                                                           .manifest_path = manifest_path,
                                                           .max_gaussians = 64,
                                                       });
    ASSERT_TRUE(sky.has_value()) << sky.error();
    EXPECT_GT(sky->marked_pixels, 0);
    EXPECT_GT(sky->gaussian_count, 0);
    EXPECT_LE(sky->gaussian_count, 64);

    auto means = sky->point_cloud.means.cpu();
    auto acc = means.accessor<float, 2>();
    const glm::vec3 p(acc(0, 0), acc(0, 1), acc(0, 2));
    EXPECT_NEAR(glm::length(p - glm::vec3(1.0f, 2.0f, 3.0f)), 10.0f, 1e-3f);
    EXPECT_LT(p.y, 2.0f);

    std::filesystem::remove_all(output_dir);
}
