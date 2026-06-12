/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_simplify_types.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {

    class SplatData;

    struct OctreeLodBuildOptions {
        // Octree subdivision stops at this many splats; each leaf becomes one
        // interior node whose children are the splats themselves. Clamped to
        // [2, 64].
        uint32_t leaf_group_splats = 8;
        // Input opacity_raw already holds display-space alpha (Spark lodOpacity
        // encoding, can exceed 1.0). Skips the sigmoid activation.
        bool input_lod_opacity = false;
        // When set, receives one entry per output node: the visible-order input
        // splat index for leaves, or UINT32_MAX for merged interior nodes.
        std::vector<uint32_t>* leaf_input_indices = nullptr;
    };

    // Build a hierarchical LOD tree from a Morton-ordered parallel octree with
    // moment-matched interior representatives. Output contract matches
    // build_bhatt_lod: node 0 is the root, children are contiguous and follow
    // their parent, every input splat survives as a leaf bit-exactly, interior
    // alpha is integrated-alpha-conserving lodOpacity (may exceed 1), and
    // interior nodes carry weight-blended SH1-3. Orders of magnitude faster
    // than the agglomerative Bhattacharyya builder on multi-million-splat
    // buckets because all phases parallelize.
    LFS_CORE_API std::expected<std::unique_ptr<SplatData>, std::string> build_octree_lod(
        const SplatData& input,
        const OctreeLodBuildOptions& options = {},
        SplatSimplifyProgressCallback progress = {});

} // namespace lfs::core
