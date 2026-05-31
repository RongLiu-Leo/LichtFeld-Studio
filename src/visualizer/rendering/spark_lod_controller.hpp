/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/splat_data.hpp"
#include <vector>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <glm/glm.hpp>

namespace lfs::vis {

class SparkLodController {
public:
    struct LodParameters {
        size_t max_splats = 1'500'000;
        float pixel_scale_limit = 0.0001f;
        float lod_render_scale = 1.0f;
        float behind_camera_penalty = 2.0f;
        float cone_foveation = 1.0f;
        float cone_inner_degrees = 0.0f;
        float cone_outer_degrees = 0.0f;
    };

    SparkLodController();
    ~SparkLodController();

    // Attach to a SplatData that has a lod_tree
    void attach(const lfs::core::SplatData& data);
    void detach();

    // Synchronous traversal. Returns selected count.
    size_t update(const glm::mat4& view_matrix, const LodParameters& params);
    void updateAsync(const glm::mat4& view_matrix, const LodParameters& params);
    bool swapAsyncResults();
    bool hasReadyResults() const;

    // Accessors
    bool hasTree() const;
    size_t selectedCount() const;
    const std::vector<uint32_t>& selectedIndices() const;

private:
    struct LodTreeNode {
        glm::vec3 center;
        float size;
        uint32_t child_start;
        uint16_t child_count;
        uint8_t lod_level;
    };

    float computePixelScale(uint32_t node_index,
                           const glm::mat4& view_matrix,
                           const LodParameters& params) const;
    size_t traverse(const glm::mat4& view_matrix,
                    const LodParameters& params,
                    std::vector<uint32_t>& out_indices) const;
    void workerLoop(std::stop_token stop_token);

    struct WorkItem {
        glm::mat4 view_matrix;
        LodParameters params;
    };

    const lfs::core::SplatData* data_ = nullptr;
    std::vector<LodTreeNode> nodes_;
    std::vector<uint32_t> selected_indices_;
    mutable std::mutex mutex_;
    std::condition_variable_any cv_;
    std::jthread worker_;
    std::optional<WorkItem> pending_work_;
    bool ready_available_ = false;
    std::vector<uint32_t> async_indices_;
    std::vector<uint32_t> ready_swap_indices_;
    LodParameters last_params_;
};

} // namespace lfs::vis
