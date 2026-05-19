/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/events.hpp"
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
#include <cmath>
#include <format>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
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
                frozen_sky_prefix = static_cast<size_t>(sky_point_cloud.size());
                frozen_sky_initial_scale = skyInitialScale(point_cloud_to_use, sky_point_cloud);

                if (max_cap > 0) {
                    const size_t cap = static_cast<size_t>(max_cap);
                    if (frozen_sky_prefix >= cap) {
                        const size_t sky_keep =
                            regular_point_count > 0 ? std::max<size_t>(1, cap / 5) : cap;
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
