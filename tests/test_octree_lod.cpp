/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/bhatt_lod.hpp"
#include "core/octree_lod.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    constexpr int kShDegree = 2;
    constexpr int kRestCoeffs = 8;

    SplatData make_synthetic_input(const std::size_t count) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> cluster_dist(-100.0f, 100.0f);
        std::normal_distribution<float> offset_dist(0.0f, 2.0f);
        std::uniform_real_distribution<float> color_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> logit_dist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> log_scale_dist(-5.0f, -2.0f);
        std::normal_distribution<float> quat_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> sh_dist(-0.5f, 0.5f);

        constexpr std::size_t kClusters = 64;
        std::vector<std::array<float, 3>> centers(kClusters);
        for (auto& c : centers) {
            c = {cluster_dist(rng), cluster_dist(rng), cluster_dist(rng) * 0.1f};
        }

        std::vector<float> means(count * 3);
        std::vector<float> sh0(count * 3);
        std::vector<float> shN(count * kRestCoeffs * 3);
        std::vector<float> scaling(count * 3);
        std::vector<float> rotation(count * 4);
        std::vector<float> opacity(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& c = centers[i % kClusters];
            means[i * 3 + 0] = c[0] + offset_dist(rng);
            means[i * 3 + 1] = c[1] + offset_dist(rng);
            means[i * 3 + 2] = c[2] + offset_dist(rng);
            for (int d = 0; d < 3; ++d) {
                sh0[i * 3 + d] = color_dist(rng);
                scaling[i * 3 + d] = log_scale_dist(rng);
            }
            for (int k = 0; k < kRestCoeffs * 3; ++k) {
                shN[i * kRestCoeffs * 3 + k] = sh_dist(rng);
            }
            float q[4] = {quat_dist(rng), quat_dist(rng), quat_dist(rng), quat_dist(rng)};
            const float norm = std::max(
                std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), 1e-6f);
            for (int d = 0; d < 4; ++d) {
                rotation[i * 4 + d] = q[d] / norm;
            }
            opacity[i] = logit_dist(rng);
        }

        return SplatData(
            kShDegree,
            Tensor::from_vector(means, {count, 3}, Device::CPU),
            Tensor::from_vector(sh0, {count, 1, 3}, Device::CPU),
            Tensor::from_vector(shN, {count, kRestCoeffs, 3}, Device::CPU),
            Tensor::from_vector(scaling, {count, 3}, Device::CPU),
            Tensor::from_vector(rotation, {count, 4}, Device::CPU),
            Tensor::from_vector(opacity, {count, 1}, Device::CPU),
            1.0f);
    }

    float ellipsoid_area(const float sx, const float sy, const float sz) {
        constexpr float p = 1.6075f;
        const float t1 = std::pow(sx * sy, p);
        const float t2 = std::pow(sx * sz, p);
        const float t3 = std::pow(sy * sz, p);
        return 4.0f * 3.14159265f * std::pow((t1 + t2 + t3) / 3.0f, 1.0f / p);
    }

    struct TreeView {
        std::size_t n = 0;
        const float* means = nullptr;
        const float* opacity = nullptr;
        const float* scaling = nullptr;
        const float* shN = nullptr;
        const lfs::core::SplatLodTree* tree = nullptr;
        Tensor means_cpu, opacity_cpu, scaling_cpu, shN_cpu;

        explicit TreeView(const SplatData& data) {
            n = static_cast<std::size_t>(data.size());
            means_cpu = data.means_raw().cpu().contiguous();
            opacity_cpu = data.opacity_raw().cpu().contiguous();
            scaling_cpu = data.scaling_raw().cpu().contiguous();
            means = means_cpu.ptr<float>();
            opacity = opacity_cpu.ptr<float>();
            scaling = scaling_cpu.ptr<float>();
            if (data.shN_raw().is_valid() && data.shN_raw().numel() > 0) {
                shN_cpu = data.shN_canonical_cpu();
                shN = shN_cpu.ptr<float>();
            }
            tree = data.lod_tree.get();
        }

        float integrated_alpha(const std::size_t i) const {
            const float sx = std::exp(scaling[i * 3 + 0]);
            const float sy = std::exp(scaling[i * 3 + 1]);
            const float sz = std::exp(scaling[i * 3 + 2]);
            return opacity[i] * ellipsoid_area(sx, sy, sz);
        }
    };

    // Shared structural contract of both builders: root at 0, children
    // contiguous and after their parent, exactly one parent per non-root node.
    void check_structure(const TreeView& v, std::size_t& leaf_count) {
        ASSERT_NE(v.tree, nullptr);
        ASSERT_EQ(v.tree->total_nodes(), v.n);
        leaf_count = 0;
        std::vector<std::uint32_t> parent_count(v.n, 0);
        for (std::size_t i = 0; i < v.n; ++i) {
            const std::uint32_t cc = v.tree->child_count[i];
            const std::uint32_t cs = v.tree->child_start[i];
            if (cc == 0) {
                ++leaf_count;
                continue;
            }
            ASSERT_GT(cs, i) << "children must follow their parent (node " << i << ")";
            ASSERT_LE(static_cast<std::size_t>(cs) + cc, v.n);
            for (std::uint32_t c = 0; c < cc; ++c) {
                ++parent_count[cs + c];
            }
        }
        ASSERT_EQ(parent_count[0], 0u) << "root must have no parent";
        for (std::size_t i = 1; i < v.n; ++i) {
            ASSERT_EQ(parent_count[i], 1u) << "node " << i << " must have exactly one parent";
        }
    }

    // Interior alpha is lodOpacity: alpha * area of a parent matches the sum
    // over its children up to per-level float rounding and the [1e-6, 1000]
    // clamp.
    void check_alpha_conservation(const TreeView& v) {
        std::size_t checked = 0;
        for (std::size_t i = 0; i < v.n; ++i) {
            const std::uint32_t cc = v.tree->child_count[i];
            if (cc == 0 || v.opacity[i] >= 999.0f) {
                continue;
            }
            const std::uint32_t cs = v.tree->child_start[i];
            float child_sum = 0.0f;
            for (std::uint32_t c = 0; c < cc; ++c) {
                child_sum += v.integrated_alpha(cs + c);
            }
            const float own = v.integrated_alpha(i);
            ASSERT_NEAR(own, child_sum, std::max(child_sum * 5e-3f, 1e-6f))
                << "integrated alpha not conserved at node " << i;
            ++checked;
        }
        ASSERT_GT(checked, 0u);
    }

    std::vector<std::array<float, 3>> sorted_leaf_positions(const TreeView& v) {
        std::vector<std::array<float, 3>> positions;
        for (std::size_t i = 0; i < v.n; ++i) {
            if (v.tree->child_count[i] == 0) {
                positions.push_back({v.means[i * 3 + 0], v.means[i * 3 + 1], v.means[i * 3 + 2]});
            }
        }
        std::sort(positions.begin(), positions.end());
        return positions;
    }

} // namespace

TEST(OctreeLod, MatchesBhattContractOnSyntheticInput) {
    constexpr std::size_t kSplats = 50'000;
    const SplatData input = make_synthetic_input(kSplats);

    auto octree = lfs::core::build_octree_lod(input);
    ASSERT_TRUE(octree.has_value()) << octree.error();
    auto bhatt = lfs::core::build_bhatt_lod(input);
    ASSERT_TRUE(bhatt.has_value()) << bhatt.error();

    const TreeView ov(**octree);
    const TreeView bv(**bhatt);

    std::size_t octree_leaves = 0;
    std::size_t bhatt_leaves = 0;
    check_structure(ov, octree_leaves);
    check_structure(bv, bhatt_leaves);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(octree_leaves, kSplats);
    EXPECT_EQ(bhatt_leaves, kSplats);

    // Exact leaf preservation: identical position multisets, identical to the
    // input.
    const auto input_means = input.means_raw().cpu().contiguous();
    const float* const in_ptr = input_means.ptr<float>();
    std::vector<std::array<float, 3>> expected(kSplats);
    for (std::size_t i = 0; i < kSplats; ++i) {
        expected[i] = {in_ptr[i * 3 + 0], in_ptr[i * 3 + 1], in_ptr[i * 3 + 2]};
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(sorted_leaf_positions(ov), expected);
    EXPECT_EQ(sorted_leaf_positions(bv), expected);

    // BFS level order is part of the octree builder's contract (bhatt emits
    // DFS pre-order and relies on the converter's relabel pass).
    for (std::size_t i = 1; i < ov.n; ++i) {
        ASSERT_GE(ov.tree->lod_level[i], ov.tree->lod_level[i - 1])
            << "octree output not level-ordered at node " << i;
    }

    check_alpha_conservation(ov);
    check_alpha_conservation(bv);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }

    // Conservation is transitive, so both roots integrate to the same total
    // leaf alpha regardless of tree shape.
    const float octree_root = ov.integrated_alpha(0);
    const float bhatt_root = bv.integrated_alpha(0);
    EXPECT_NEAR(octree_root, bhatt_root, bhatt_root * 0.02f);

    // Interior nodes carry blended SH1-3: every coefficient is a convex
    // combination of its children's, and the blend is non-trivial.
    constexpr std::size_t kShFloats = kRestCoeffs * 3;
    float max_abs_interior_sh = 0.0f;
    for (std::size_t i = 0; i < ov.n; ++i) {
        const std::uint32_t cc = ov.tree->child_count[i];
        if (cc == 0) {
            continue;
        }
        const std::uint32_t cs = ov.tree->child_start[i];
        for (std::size_t k = 0; k < kShFloats; ++k) {
            float lo = std::numeric_limits<float>::max();
            float hi = std::numeric_limits<float>::lowest();
            for (std::uint32_t c = 0; c < cc; ++c) {
                const float val = ov.shN[(cs + c) * kShFloats + k];
                lo = std::min(lo, val);
                hi = std::max(hi, val);
            }
            const float own = ov.shN[i * kShFloats + k];
            ASSERT_GE(own, lo - 1e-4f) << "SH blend out of range at node " << i << " coeff " << k;
            ASSERT_LE(own, hi + 1e-4f) << "SH blend out of range at node " << i << " coeff " << k;
            max_abs_interior_sh = std::max(max_abs_interior_sh, std::abs(own));
        }
    }
    EXPECT_GT(max_abs_interior_sh, 1e-3f) << "interior SH must not collapse to zero";

    // The octree tree is strictly more compact than the binary merge tree.
    EXPECT_LT(ov.n, bv.n);
}

TEST(OctreeLod, SmallInputs) {
    for (const std::size_t count : {std::size_t{1}, std::size_t{2}, std::size_t{7},
                                    std::size_t{9}, std::size_t{100}}) {
        const SplatData input = make_synthetic_input(count);
        auto octree = lfs::core::build_octree_lod(input);
        ASSERT_TRUE(octree.has_value()) << octree.error();

        const TreeView v(**octree);
        std::size_t leaves = 0;
        check_structure(v, leaves);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        EXPECT_EQ(leaves, count) << "count=" << count;
        if (count == 1) {
            EXPECT_EQ(v.n, 1u);
        } else {
            check_alpha_conservation(v);
        }
    }
}

TEST(OctreeLod, IdenticalPositionsStayBounded) {
    constexpr std::size_t kSplats = 1'000;
    std::vector<float> means(kSplats * 3, 1.5f);
    std::vector<float> sh0(kSplats * 3, 0.2f);
    std::vector<float> scaling(kSplats * 3, -3.0f);
    std::vector<float> rotation(kSplats * 4, 0.0f);
    std::vector<float> opacity(kSplats, 0.5f);
    for (std::size_t i = 0; i < kSplats; ++i) {
        rotation[i * 4] = 1.0f;
    }
    const SplatData input(
        0,
        Tensor::from_vector(means, {kSplats, 3}, Device::CPU),
        Tensor::from_vector(sh0, {kSplats, 1, 3}, Device::CPU),
        Tensor{},
        Tensor::from_vector(scaling, {kSplats, 3}, Device::CPU),
        Tensor::from_vector(rotation, {kSplats, 4}, Device::CPU),
        Tensor::from_vector(opacity, {kSplats, 1}, Device::CPU),
        1.0f);

    auto octree = lfs::core::build_octree_lod(input);
    ASSERT_TRUE(octree.has_value()) << octree.error();
    const TreeView v(**octree);
    std::size_t leaves = 0;
    check_structure(v, leaves);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(leaves, kSplats);
    std::uint16_t max_children = 0;
    for (std::size_t i = 0; i < v.n; ++i) {
        max_children = std::max(max_children, v.tree->child_count[i]);
    }
    EXPECT_LE(max_children, 64);
}
