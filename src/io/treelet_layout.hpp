/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Treelet band layout shared by the PLY converter and the RAD re-layout
// migrator: global order is band-major (a coarse cut stays a file prefix);
// within a band, nodes group per band-root treelet — roots first in
// sibling-contiguous order, then each root's in-band descendants in BFS.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace lfs::io::treelet {

    inline constexpr std::size_t kTreeletBandLevels = 4;

    // Band-BFS ranks over one subtree's local level layout. Nodes of each
    // global band [k*B, (k+1)*B) get contiguous in-block ranks: every
    // band-first-level node roots a treelet and its in-band descendants
    // follow in BFS order, so sibling groups stay contiguous. `depth`
    // aligns local levels to global bands; levels below `min_local` are
    // not emitted (multi-bucket roots live in the top tree).
    struct TreeletRanks {
        std::vector<std::uint32_t> rank;       // [local] in-block rank
        std::vector<std::uint64_t> band_count; // nodes per global band
    };
    [[nodiscard]] inline TreeletRanks build_treelet_ranks(
        const std::vector<std::uint32_t>& level_starts,
        const std::vector<std::uint16_t>& child_count,
        const std::vector<std::uint32_t>& child_start,
        const std::size_t depth,
        const std::size_t min_local,
        const std::size_t global_bands) {
        TreeletRanks out;
        const std::size_t levels = level_starts.size() - 1;
        const std::size_t total = level_starts.back();
        out.rank.assign(total, 0u);
        out.band_count.assign(global_bands, 0u);
        std::vector<std::uint32_t> queue;
        for (std::size_t band = 0; band < global_bands; ++band) {
            const std::size_t g_lo = band * kTreeletBandLevels;
            const std::size_t g_hi = g_lo + kTreeletBandLevels;
            const std::size_t l_lo = std::max<std::size_t>(g_lo > depth ? g_lo - depth : 0, min_local);
            if (l_lo >= levels || depth + l_lo >= g_hi) {
                continue;
            }
            const std::size_t l_hi = std::min<std::size_t>(levels, g_hi - depth);
            // Roots first, in old (sibling-contiguous) order: cross-band
            // child groups point at roots and must stay contiguous. Each
            // root's in-band descendants follow treelet-by-treelet.
            std::uint32_t next = 0;
            for (std::uint32_t root = level_starts[l_lo]; root < level_starts[l_lo + 1]; ++root) {
                out.rank[root] = next++;
            }
            for (std::uint32_t root = level_starts[l_lo]; root < level_starts[l_lo + 1]; ++root) {
                queue.clear();
                queue.push_back(root);
                std::size_t head = 0;
                while (head < queue.size()) {
                    const std::uint32_t node = queue[head++];
                    if (node != root) {
                        out.rank[node] = next++;
                    }
                    const std::size_t node_level =
                        std::upper_bound(level_starts.begin(), level_starts.end(), node) -
                        level_starts.begin() - 1;
                    if (node_level + 1 < l_hi && child_count[node] > 0) {
                        for (std::uint32_t c = 0; c < child_count[node]; ++c) {
                            queue.push_back(child_start[node] + c);
                        }
                    }
                }
            }
            out.band_count[band] = next;
        }
        return out;
    }

} // namespace lfs::io::treelet
