/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spark_lod_controller.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>

namespace lfs::vis {

SparkLodController::SparkLodController() = default;
SparkLodController::~SparkLodController() = default;

void SparkLodController::attach(const lfs::core::SplatData& data) {
    detach();
    if (!data.lod_tree || !data.lod_tree->has_tree()) {
        return;
    }

    data_ = &data;
    const auto& tree = *data.lod_tree;
    const size_t n = tree.total_nodes();
    if (n == 0 || n > static_cast<size_t>(data.size())) {
        detach();
        return;
    }
    if (tree.child_start.size() < n || tree.child_count.size() < n) {
        detach();
        return;
    }
    nodes_.resize(n);

    const bool has_cached_centers = tree.centers.size() >= n;
    const bool has_cached_sizes = tree.sizes.size() >= n;
    const float* means_ptr = nullptr;
    const float* scales_ptr = nullptr;
    lfs::core::Tensor means_cpu;
    lfs::core::Tensor scaling_cpu;
    if (!has_cached_centers) {
        means_cpu = data.means().cpu();
        means_ptr = means_cpu.ptr<float>();
    }
    if (!has_cached_sizes) {
        scaling_cpu = data.scaling_raw().cpu();
        scales_ptr = scaling_cpu.ptr<float>();
    }

    for (size_t i = 0; i < n; ++i) {
        if (has_cached_centers) {
            nodes_[i].center = tree.centers[i];
        } else {
            nodes_[i].center = glm::vec3(
                means_ptr[i * 3 + 0],
                means_ptr[i * 3 + 1],
                means_ptr[i * 3 + 2]);
        }

        if (has_cached_sizes) {
            nodes_[i].size = tree.sizes[i];
        } else {
            float sx = std::exp(scales_ptr[i * 3 + 0]);
            float sy = std::exp(scales_ptr[i * 3 + 1]);
            float sz = std::exp(scales_ptr[i * 3 + 2]);
            nodes_[i].size = 2.0f * std::max({sx, sy, sz});
        }

        nodes_[i].child_start = tree.child_start[i];
        nodes_[i].child_count = tree.child_count[i];
        nodes_[i].lod_level = (i < tree.lod_level.size()) ? tree.lod_level[i] : 0;
    }

    // Compute lod_level via BFS if not provided by loader
    if (tree.lod_level.empty()) {
        std::vector<uint8_t> bfs_level(n, 0);
        std::queue<uint32_t> q;
        q.push(0);
        bfs_level[0] = 0;
        while (!q.empty()) {
            uint32_t idx = q.front(); q.pop();
            uint8_t level = bfs_level[idx];
            nodes_[idx].lod_level = level;
            for (uint32_t c = 0; c < nodes_[idx].child_count; ++c) {
                uint32_t child_idx = nodes_[idx].child_start + c;
                if (child_idx < n) {
                    bfs_level[child_idx] = level + 1;
                    q.push(child_idx);
                }
            }
        }
    }

    std::size_t non_leaf_count = 0;
    std::uint16_t max_child_count = 0;
    for (const auto& node : nodes_) {
        if (node.child_count > 0) {
            ++non_leaf_count;
            max_child_count = std::max(max_child_count, node.child_count);
        }
    }
    LOG_INFO(
        "LOD attach: nodes={} non_leaf_nodes={} root_child_count={} max_child_count={}",
        nodes_.size(),
        non_leaf_count,
        nodes_.empty() ? 0u : static_cast<unsigned>(nodes_[0].child_count),
        static_cast<unsigned>(max_child_count));
}

void SparkLodController::detach() {
    data_ = nullptr;
    nodes_.clear();
    selected_indices_.clear();
    selected_lod_levels_.clear();
}

float SparkLodController::computePixelScale(uint32_t node_index,
                                             const glm::mat4& view_matrix,
                                             const LodParameters& params) const {
    const auto& node = nodes_[node_index];
    glm::vec4 center_vs = view_matrix * glm::vec4(node.center, 1.0f);
    float radial_dist = glm::length(glm::vec3(center_vs));
    if (radial_dist <= 0.0f) {
        return std::numeric_limits<float>::max();
    }

    float pixel_scale = (node.size * params.lod_render_scale) / radial_dist;

    if (center_vs.z < 0.0f) {
        pixel_scale *= params.behind_camera_penalty;
    }

    return pixel_scale;
}

size_t SparkLodController::update(const glm::mat4& view_matrix, const LodParameters& params) {
    selected_indices_.clear();
    selected_lod_levels_.clear();
    if (nodes_.empty() || params.max_splats == 0) {
        return 0;
    }

    selected_indices_.reserve(params.max_splats);
    selected_lod_levels_.reserve(params.max_splats);

    struct HeapNode {
        uint32_t index;
        float pixel_scale;
    };

    struct HeapCompare {
        bool operator()(const HeapNode& a, const HeapNode& b) const {
            return a.pixel_scale < b.pixel_scale; // max-heap: larger scale first
        }
    };

    std::priority_queue<HeapNode, std::vector<HeapNode>, HeapCompare> heap;
    std::vector<std::uint8_t> queued(nodes_.size(), 0);

    // Seed with root node
    heap.push({0, computePixelScale(0, view_matrix, params)});
    queued[0] = 1;

    // Matches Spark semantics: this tracks output size after draining frontier.
    size_t num_splats = 1;

    while (!heap.empty()) {
        const auto top = heap.top();
        if (top.pixel_scale <= params.pixel_scale_limit) {
            break;
        }

        heap.pop();
        const auto& node = nodes_[top.index];

        if (node.child_count == 0) {
            // Leaf: output directly.
            selected_indices_.push_back(top.index);
            selected_lod_levels_.push_back(nodes_[top.index].lod_level);
            if (selected_indices_.size() >= params.max_splats) {
                break;
            }
        } else {
            // Internal node: check budget before expanding.
            const size_t new_num_splats = num_splats - 1 + static_cast<size_t>(node.child_count);
            if (new_num_splats > params.max_splats) {
                // Keep this node in the frontier output (Spark behavior).
                heap.push(top);
                break;
            }

            // Expand children. Children already below threshold go directly to output.
            for (uint32_t c = 0; c < node.child_count; ++c) {
                const uint32_t child_idx = node.child_start + c;
                if (child_idx < nodes_.size() && !queued[child_idx]) {
                    queued[child_idx] = 1;
                    const float scale = computePixelScale(child_idx, view_matrix, params);
                    if (scale <= params.pixel_scale_limit) {
                        selected_indices_.push_back(child_idx);
                        selected_lod_levels_.push_back(nodes_[child_idx].lod_level);
                    } else {
                        heap.push({child_idx, scale});
                    }
                }
            }
            num_splats = new_num_splats;
            if (selected_indices_.size() >= params.max_splats) {
                break;
            }
        }
    }

    // Drain remaining frontier nodes while honoring the budget.
    while (!heap.empty() && selected_indices_.size() < params.max_splats) {
        selected_indices_.push_back(heap.top().index);
        selected_lod_levels_.push_back(nodes_[heap.top().index].lod_level);
        heap.pop();
    }

    last_params_ = params;
    return selected_indices_.size();
}

bool SparkLodController::hasTree() const {
    return !nodes_.empty();
}

size_t SparkLodController::selectedCount() const {
    return selected_indices_.size();
}

const std::vector<uint32_t>& SparkLodController::selectedIndices() const {
    return selected_indices_;
}

const std::vector<uint32_t>& SparkLodController::selectedLodLevels() const {
    return selected_lod_levels_;
}

} // namespace lfs::vis
