/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/octree_lod.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "lod_merge_math.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace lfs::core {

    namespace {

        using namespace lodmath;

        constexpr int kMortonLevels = 21;
        constexpr uint32_t kMaxGroupSplats = 64;
        constexpr uint32_t kInteriorFlag = 0x80000000u;

        // Interleaved display-space splat arrays (visible splats only).
        struct SplatSoa {
            std::vector<float> mean;  // [N,3]
            std::vector<float> scale; // [N,3] linear
            std::vector<float> quat;  // [N,4] normalized (w,x,y,z)
            std::vector<float> alpha; // [N] display-space (lodOpacity domain)
            std::vector<float> rgb;   // [N,3] display-space color
            std::vector<float> sh1;   // [N,9]
            std::vector<float> sh2;   // [N,15]
            std::vector<float> sh3;   // [N,21]
            int max_sh_degree = 0;
            size_t count = 0;
        };

        SplatSoa extract_soa(const SplatData& input, const bool input_lod_opacity) {
            const auto means_cpu = input.means_raw().cpu().contiguous();
            const auto scaling_cpu = input.scaling_raw().cpu().contiguous();
            const auto rotation_cpu = input.rotation_raw().cpu().contiguous();
            const auto opacity_cpu = input.opacity_raw().cpu().contiguous();
            const auto sh0_cpu = input.sh0_raw().cpu().contiguous();
            const float* const means_ptr = means_cpu.ptr<float>();
            const float* const scaling_ptr = scaling_cpu.ptr<float>();
            const float* const rotation_ptr = rotation_cpu.ptr<float>();
            const float* const opacity_ptr = opacity_cpu.ptr<float>();
            const float* const sh0_ptr = sh0_cpu.ptr<float>();

            const size_t total = input.size();
            const bool has_deleted = input.has_deleted_mask() && input.deleted().count_nonzero() > 0;
            const auto deleted_cpu = has_deleted ? input.deleted().cpu().contiguous() : Tensor{};
            const uint8_t* const deleted_ptr = has_deleted ? deleted_cpu.ptr<uint8_t>() : nullptr;

            std::vector<uint32_t> visible;
            if (has_deleted) {
                visible.reserve(total);
                for (size_t i = 0; i < total; ++i) {
                    if (deleted_ptr[i] == 0) {
                        visible.push_back(static_cast<uint32_t>(i));
                    }
                }
            }
            const size_t n = has_deleted ? visible.size() : total;

            SplatSoa soa;
            soa.count = n;
            soa.max_sh_degree = input.get_max_sh_degree();
            soa.mean.resize(n * 3);
            soa.scale.resize(n * 3);
            soa.quat.resize(n * 4);
            soa.alpha.resize(n);
            soa.rgb.resize(n * 3);
            if (soa.max_sh_degree >= 1) {
                soa.sh1.resize(n * 9, 0.0f);
            }
            if (soa.max_sh_degree >= 2) {
                soa.sh2.resize(n * 15, 0.0f);
            }
            if (soa.max_sh_degree >= 3) {
                soa.sh3.resize(n * 21, 0.0f);
            }

            Tensor shN_canonical;
            const float* shN_ptr = nullptr;
            int shN_rest_coeffs = 0;
            if (input.shN_raw().is_valid() && input.shN_raw().numel() > 0 && input.max_sh_coeffs_rest() > 0) {
                shN_canonical = input.shN_canonical_cpu();
                shN_ptr = shN_canonical.ptr<float>();
                shN_rest_coeffs = static_cast<int>(input.max_sh_coeffs_rest());
            }

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const size_t src = has_deleted ? visible[i] : i;
                        const size_t s3 = src * 3;
                        const size_t s4 = src * 4;

                        soa.mean[i * 3 + 0] = means_ptr[s3 + 0];
                        soa.mean[i * 3 + 1] = means_ptr[s3 + 1];
                        soa.mean[i * 3 + 2] = means_ptr[s3 + 2];
                        soa.scale[i * 3 + 0] = activated_scale(scaling_ptr[s3 + 0]);
                        soa.scale[i * 3 + 1] = activated_scale(scaling_ptr[s3 + 1]);
                        soa.scale[i * 3 + 2] = activated_scale(scaling_ptr[s3 + 2]);

                        float qw = rotation_ptr[s4 + 0];
                        float qx = rotation_ptr[s4 + 1];
                        float qy = rotation_ptr[s4 + 2];
                        float qz = rotation_ptr[s4 + 3];
                        const float inv_q =
                            1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
                        soa.quat[i * 4 + 0] = qw * inv_q;
                        soa.quat[i * 4 + 1] = qx * inv_q;
                        soa.quat[i * 4 + 2] = qy * inv_q;
                        soa.quat[i * 4 + 3] = qz * inv_q;

                        soa.alpha[i] = input_lod_opacity ? std::max(opacity_ptr[src], 0.0f)
                                                         : sigmoid(opacity_ptr[src]);
                        soa.rgb[i * 3 + 0] = 0.5f + SH_C0 * sh0_ptr[s3 + 0];
                        soa.rgb[i * 3 + 1] = 0.5f + SH_C0 * sh0_ptr[s3 + 1];
                        soa.rgb[i * 3 + 2] = 0.5f + SH_C0 * sh0_ptr[s3 + 2];

                        if (shN_ptr != nullptr) {
                            const float* const rest = shN_ptr + src * static_cast<size_t>(shN_rest_coeffs) * 3;
                            if (soa.max_sh_degree >= 1 && shN_rest_coeffs >= 3) {
                                std::copy_n(rest, 9, soa.sh1.data() + i * 9);
                            }
                            if (soa.max_sh_degree >= 2 && shN_rest_coeffs >= 8) {
                                std::copy_n(rest + 9, 15, soa.sh2.data() + i * 15);
                            }
                            if (soa.max_sh_degree >= 3 && shN_rest_coeffs >= 15) {
                                std::copy_n(rest + 24, 21, soa.sh3.data() + i * 21);
                            }
                        }
                    }
                });
            return soa;
        }

        // ----------------------------------------------------------------
        // Morton ordering
        // ----------------------------------------------------------------

        inline uint64_t expand_bits_21(uint64_t v) {
            v &= 0x1fffff;
            v = (v | v << 32) & 0x1f00000000ffffULL;
            v = (v | v << 16) & 0x1f0000ff0000ffULL;
            v = (v | v << 8) & 0x100f00f00f00f00fULL;
            v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
            v = (v | v << 2) & 0x1249249249249249ULL;
            return v;
        }

        inline uint64_t morton_code(const float* p, const float* mn, const float* inv_extent) {
            uint64_t axes[3];
            for (int a = 0; a < 3; ++a) {
                const float norm = (p[a] - mn[a]) * inv_extent[a];
                const float f = std::isfinite(norm)
                                    ? std::clamp(norm, 0.0f, 1.0f) * static_cast<float>((1 << kMortonLevels) - 1)
                                    : 0.0f;
                axes[a] = static_cast<uint64_t>(f);
            }
            return (expand_bits_21(axes[0]) << 2) | (expand_bits_21(axes[1]) << 1) | expand_bits_21(axes[2]);
        }

        struct MortonEntry {
            uint64_t code;
            uint32_t index;
        };

        // ----------------------------------------------------------------
        // Octree topology over the sorted code range
        // ----------------------------------------------------------------

        struct OctNode {
            uint32_t begin = 0;
            uint32_t end = 0;
            uint32_t children[8] = {};
            uint8_t child_count = 0;
            uint8_t depth = 0;
            bool leaf = false;

            [[nodiscard]] uint32_t splat_count() const { return end - begin; }
        };

        struct Topology {
            const std::vector<uint64_t>& codes;
            uint32_t group_max;
            std::vector<OctNode> nodes;

            static int octant(const uint64_t code, const int level) {
                return static_cast<int>((code >> (3 * level)) & 7u);
            }

            uint32_t build(const uint32_t begin, const uint32_t end, int level, const int depth) {
                assert(depth < 250);
                const uint32_t idx = static_cast<uint32_t>(nodes.size());
                nodes.emplace_back();
                nodes[idx].begin = begin;
                nodes[idx].end = end;
                nodes[idx].depth = static_cast<uint8_t>(depth);

                const uint32_t count = end - begin;
                if (count <= group_max) {
                    nodes[idx].leaf = true;
                    return idx;
                }

                // Skip levels where the whole range shares one octant digit; the
                // range is sorted, so first == last implies all equal.
                while (level >= 0 && octant(codes[begin], level) == octant(codes[end - 1], level)) {
                    --level;
                }

                uint32_t kids[8];
                uint8_t kid_count = 0;
                if (level < 0) {
                    // Identical codes (or Morton resolution exhausted): split the
                    // run into near-equal eighths so groups stay bounded.
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t b = begin + static_cast<uint32_t>(
                                                       (static_cast<uint64_t>(count) * k) / 8);
                        const uint32_t e = begin + static_cast<uint32_t>(
                                                       (static_cast<uint64_t>(count) * (k + 1)) / 8);
                        if (e > b) {
                            kids[kid_count++] = build(b, e, level, depth + 1);
                        }
                    }
                } else {
                    uint32_t child_begin = begin;
                    for (int oct = 0; oct < 8 && child_begin < end; ++oct) {
                        const uint64_t* const lo = codes.data() + child_begin;
                        const uint64_t* const hi = codes.data() + end;
                        const int shift = 3 * level;
                        const uint64_t* const it = std::upper_bound(
                            lo, hi, static_cast<uint64_t>(oct),
                            [shift](const uint64_t v, const uint64_t c) {
                                return v < ((c >> shift) & 7u);
                            });
                        const uint32_t child_end = static_cast<uint32_t>(it - codes.data());
                        if (child_end > child_begin) {
                            kids[kid_count++] = build(child_begin, child_end, level - 1, depth + 1);
                        }
                        child_begin = child_end;
                    }
                }

                OctNode& node = nodes[idx]; // re-fetch: recursion may reallocate
                node.child_count = kid_count;
                std::copy_n(kids, kid_count, node.children);
                assert(kid_count >= 2);
                return idx;
            }
        };

        // ----------------------------------------------------------------
        // Moment-matched representatives
        // ----------------------------------------------------------------

        struct Reps {
            std::vector<float> mean;  // [M,3]
            std::vector<float> scale; // [M,3]
            std::vector<float> quat;  // [M,4]
            std::vector<float> alpha; // [M]
            std::vector<float> rgb;   // [M,3]
            std::vector<float> area;  // [M]
            std::vector<float> sh1, sh2, sh3;
        };

        struct MergeInput {
            uint32_t count = 0;
            const float* mean[kMaxGroupSplats];
            const float* scale[kMaxGroupSplats];
            const float* quat[kMaxGroupSplats];
            const float* rgb[kMaxGroupSplats];
            const float* sh1[kMaxGroupSplats];
            const float* sh2[kMaxGroupSplats];
            const float* sh3[kMaxGroupSplats];
            float alpha[kMaxGroupSplats];
            float area[kMaxGroupSplats];
        };

        // Gaussian mixture moment matching, generalizing the pairwise merge in
        // bhatt_lod.cpp to n children: weights w_i = alpha_i * area_i, merged
        // mean/covariance are the mixture moments (covariance includes the
        // spread of child means), opacity conserves the cluster's integrated
        // alpha (lodOpacity, may exceed 1), and color + SH1-3 blend with the
        // same weights.
        void merge_group(const MergeInput& in, const int max_sh_degree, Reps& out, const size_t dst) {
            double weight_sum = 0.0;
            double mu[3] = {0.0, 0.0, 0.0};
            double rgb_acc[3] = {0.0, 0.0, 0.0};
            float w[kMaxGroupSplats];
            for (uint32_t i = 0; i < in.count; ++i) {
                w[i] = std::max(in.alpha[i] * in.area[i], 1e-30f);
                weight_sum += w[i];
                for (int d = 0; d < 3; ++d) {
                    mu[d] += static_cast<double>(w[i]) * in.mean[i][d];
                    rgb_acc[d] += static_cast<double>(w[i]) * in.rgb[i][d];
                }
            }
            const double inv_w = 1.0 / weight_sum;
            for (int d = 0; d < 3; ++d) {
                mu[d] *= inv_w;
                rgb_acc[d] *= inv_w;
            }

            double cov[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (uint32_t i = 0; i < in.count; ++i) {
                float cxx, cxy, cxz, cyy, cyz, czz, cdet;
                compute_covariance_from_scale_quat(
                    in.scale[i][0], in.scale[i][1], in.scale[i][2],
                    in.quat[i][0], in.quat[i][1], in.quat[i][2], in.quat[i][3],
                    cxx, cxy, cxz, cyy, cyz, czz, cdet);
                const double dx = in.mean[i][0] - mu[0];
                const double dy = in.mean[i][1] - mu[1];
                const double dz = in.mean[i][2] - mu[2];
                const double wi = w[i];
                cov[0] += wi * (cxx + dx * dx);
                cov[1] += wi * (cxy + dx * dy);
                cov[2] += wi * (cxz + dx * dz);
                cov[3] += wi * (cyy + dy * dy);
                cov[4] += wi * (cyz + dy * dz);
                cov[5] += wi * (czz + dz * dz);
            }
            const std::array<float, 9> sigma = {
                static_cast<float>(cov[0] * inv_w) + kEpsCov,
                static_cast<float>(cov[1] * inv_w),
                static_cast<float>(cov[2] * inv_w),
                static_cast<float>(cov[1] * inv_w),
                static_cast<float>(cov[3] * inv_w) + kEpsCov,
                static_cast<float>(cov[4] * inv_w),
                static_cast<float>(cov[2] * inv_w),
                static_cast<float>(cov[4] * inv_w),
                static_cast<float>(cov[5] * inv_w) + kEpsCov,
            };

            const auto eig = eigen_symmetric_3x3_jacobi(sigma);
            const float sx = std::sqrt(std::max(eig.values[0], kMinEval));
            const float sy = std::sqrt(std::max(eig.values[1], kMinEval));
            const float sz = std::sqrt(std::max(eig.values[2], kMinEval));
            std::array<float, 4> q;
            rotmat_to_quat(eig.vectors, q);

            const float merged_area = ellipsoid_area(sx, sy, sz);
            const float merged_alpha = std::clamp(
                static_cast<float>(weight_sum) / std::max(merged_area, 1e-30f), 0.000001f, 1000.0f);

            out.mean[dst * 3 + 0] = static_cast<float>(mu[0]);
            out.mean[dst * 3 + 1] = static_cast<float>(mu[1]);
            out.mean[dst * 3 + 2] = static_cast<float>(mu[2]);
            out.scale[dst * 3 + 0] = sx;
            out.scale[dst * 3 + 1] = sy;
            out.scale[dst * 3 + 2] = sz;
            std::copy_n(q.data(), 4, out.quat.data() + dst * 4);
            out.alpha[dst] = merged_alpha;
            out.rgb[dst * 3 + 0] = static_cast<float>(rgb_acc[0]);
            out.rgb[dst * 3 + 1] = static_cast<float>(rgb_acc[1]);
            out.rgb[dst * 3 + 2] = static_cast<float>(rgb_acc[2]);
            out.area[dst] = merged_area;

            const auto blend_sh = [&](const float* const* src, std::vector<float>& dst_vec, const int coeffs) {
                double acc[21];
                std::fill_n(acc, coeffs, 0.0);
                for (uint32_t i = 0; i < in.count; ++i) {
                    for (int k = 0; k < coeffs; ++k) {
                        acc[k] += static_cast<double>(w[i]) * src[i][k];
                    }
                }
                float* const d = dst_vec.data() + dst * static_cast<size_t>(coeffs);
                for (int k = 0; k < coeffs; ++k) {
                    d[k] = static_cast<float>(acc[k] * inv_w);
                }
            };
            if (max_sh_degree >= 1) {
                blend_sh(in.sh1, out.sh1, 9);
            }
            if (max_sh_degree >= 2) {
                blend_sh(in.sh2, out.sh2, 15);
            }
            if (max_sh_degree >= 3) {
                blend_sh(in.sh3, out.sh3, 21);
            }
        }

    } // namespace

    std::expected<std::unique_ptr<SplatData>, std::string> build_octree_lod(
        const SplatData& input,
        const OctreeLodBuildOptions& options,
        SplatSimplifyProgressCallback progress) {
        try {
            const auto t_start = std::chrono::high_resolution_clock::now();
            const auto elapsed_ms = [&t_start] {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - t_start)
                    .count();
            };

            if (!input.means_raw().is_valid() || input.size() == 0) {
                return std::unexpected("build_octree_lod: input splat is empty");
            }
            const uint32_t group_max = std::clamp(options.leaf_group_splats, 2u, kMaxGroupSplats);

            const SplatSoa soa = extract_soa(input, options.input_lod_opacity);
            const size_t n = soa.count;
            if (n == 0) {
                return std::unexpected("build_octree_lod: no visible gaussians");
            }
            const auto t_extract = elapsed_ms();
            if (progress && !progress(0.05f, "Extracting splats")) {
                return std::unexpected("build_octree_lod: cancelled by user");
            }

            // Morton codes over the bucket bounds, sorted with index tie-break
            // for determinism.
            float mn[3] = {std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
            float mx[3] = {std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest()};
            for (size_t i = 0; i < n; ++i) {
                for (int a = 0; a < 3; ++a) {
                    const float v = soa.mean[i * 3 + a];
                    if (std::isfinite(v)) {
                        mn[a] = std::min(mn[a], v);
                        mx[a] = std::max(mx[a], v);
                    }
                }
            }
            float inv_extent[3];
            for (int a = 0; a < 3; ++a) {
                const float extent = mx[a] - mn[a];
                inv_extent[a] = extent > 0.0f ? 1.0f / extent : 0.0f;
            }

            std::vector<MortonEntry> entries(n);
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        entries[i] = {morton_code(soa.mean.data() + i * 3, mn, inv_extent),
                                      static_cast<uint32_t>(i)};
                    }
                });
            tbb::parallel_sort(entries.begin(), entries.end(),
                               [](const MortonEntry& a, const MortonEntry& b) {
                                   return a.code != b.code ? a.code < b.code : a.index < b.index;
                               });
            std::vector<uint64_t> codes(n);
            std::vector<uint32_t> order(n);
            for (size_t i = 0; i < n; ++i) {
                codes[i] = entries[i].code;
                order[i] = entries[i].index;
            }
            entries.clear();
            entries.shrink_to_fit();
            const auto t_sort = elapsed_ms();
            if (progress && !progress(0.25f, "Morton sort")) {
                return std::unexpected("build_octree_lod: cancelled by user");
            }

            Topology topo{codes, group_max, {}};
            topo.nodes.reserve(2 * n / group_max + 64);
            topo.build(0, static_cast<uint32_t>(n), kMortonLevels - 1, 0);
            const size_t node_count = topo.nodes.size();
            const auto t_topo = elapsed_ms();
            if (progress && !progress(0.35f, "Octree topology")) {
                return std::unexpected("build_octree_lod: cancelled by user");
            }

            // Bottom-up moment matching, parallel within each depth. Children
            // have higher pre-order indices, so deeper levels finish first.
            Reps reps;
            reps.mean.resize(node_count * 3);
            reps.scale.resize(node_count * 3);
            reps.quat.resize(node_count * 4);
            reps.alpha.resize(node_count);
            reps.rgb.resize(node_count * 3);
            reps.area.resize(node_count);
            if (soa.max_sh_degree >= 1) {
                reps.sh1.resize(node_count * 9);
            }
            if (soa.max_sh_degree >= 2) {
                reps.sh2.resize(node_count * 15);
            }
            if (soa.max_sh_degree >= 3) {
                reps.sh3.resize(node_count * 21);
            }

            int max_depth = 0;
            for (const OctNode& node : topo.nodes) {
                max_depth = std::max(max_depth, static_cast<int>(node.depth));
            }
            std::vector<std::vector<uint32_t>> by_depth(static_cast<size_t>(max_depth) + 1);
            for (uint32_t i = 0; i < node_count; ++i) {
                by_depth[topo.nodes[i].depth].push_back(i);
            }

            const auto splat_child = [&](MergeInput& in, const uint32_t splat) {
                const uint32_t k = in.count++;
                in.mean[k] = soa.mean.data() + static_cast<size_t>(splat) * 3;
                in.scale[k] = soa.scale.data() + static_cast<size_t>(splat) * 3;
                in.quat[k] = soa.quat.data() + static_cast<size_t>(splat) * 4;
                in.rgb[k] = soa.rgb.data() + static_cast<size_t>(splat) * 3;
                in.sh1[k] = soa.max_sh_degree >= 1 ? soa.sh1.data() + static_cast<size_t>(splat) * 9 : nullptr;
                in.sh2[k] = soa.max_sh_degree >= 2 ? soa.sh2.data() + static_cast<size_t>(splat) * 15 : nullptr;
                in.sh3[k] = soa.max_sh_degree >= 3 ? soa.sh3.data() + static_cast<size_t>(splat) * 21 : nullptr;
                in.alpha[k] = soa.alpha[splat];
                in.area[k] = ellipsoid_area(in.scale[k][0], in.scale[k][1], in.scale[k][2]);
            };
            const auto rep_child = [&](MergeInput& in, const uint32_t node) {
                const uint32_t k = in.count++;
                in.mean[k] = reps.mean.data() + static_cast<size_t>(node) * 3;
                in.scale[k] = reps.scale.data() + static_cast<size_t>(node) * 3;
                in.quat[k] = reps.quat.data() + static_cast<size_t>(node) * 4;
                in.rgb[k] = reps.rgb.data() + static_cast<size_t>(node) * 3;
                in.sh1[k] = soa.max_sh_degree >= 1 ? reps.sh1.data() + static_cast<size_t>(node) * 9 : nullptr;
                in.sh2[k] = soa.max_sh_degree >= 2 ? reps.sh2.data() + static_cast<size_t>(node) * 15 : nullptr;
                in.sh3[k] = soa.max_sh_degree >= 3 ? reps.sh3.data() + static_cast<size_t>(node) * 21 : nullptr;
                in.alpha[k] = reps.alpha[node];
                in.area[k] = reps.area[node];
            };

            for (int d = max_depth; d >= 0; --d) {
                const auto& bucket = by_depth[static_cast<size_t>(d)];
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, bucket.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t bi = range.begin(); bi != range.end(); ++bi) {
                            const uint32_t ni = bucket[bi];
                            const OctNode& node = topo.nodes[ni];
                            MergeInput in;
                            if (node.leaf) {
                                if (node.splat_count() == 1) {
                                    const size_t s = order[node.begin];
                                    std::copy_n(soa.mean.data() + s * 3, 3, reps.mean.data() + ni * 3);
                                    std::copy_n(soa.scale.data() + s * 3, 3, reps.scale.data() + ni * 3);
                                    std::copy_n(soa.quat.data() + s * 4, 4, reps.quat.data() + ni * 4);
                                    reps.alpha[ni] = soa.alpha[s];
                                    std::copy_n(soa.rgb.data() + s * 3, 3, reps.rgb.data() + ni * 3);
                                    reps.area[ni] = ellipsoid_area(soa.scale[s * 3 + 0],
                                                                   soa.scale[s * 3 + 1],
                                                                   soa.scale[s * 3 + 2]);
                                    if (soa.max_sh_degree >= 1) {
                                        std::copy_n(soa.sh1.data() + s * 9, 9, reps.sh1.data() + ni * 9);
                                    }
                                    if (soa.max_sh_degree >= 2) {
                                        std::copy_n(soa.sh2.data() + s * 15, 15, reps.sh2.data() + ni * 15);
                                    }
                                    if (soa.max_sh_degree >= 3) {
                                        std::copy_n(soa.sh3.data() + s * 21, 21, reps.sh3.data() + ni * 21);
                                    }
                                    continue;
                                }
                                for (uint32_t i = node.begin; i < node.end; ++i) {
                                    splat_child(in, order[i]);
                                }
                            } else {
                                for (uint8_t c = 0; c < node.child_count; ++c) {
                                    rep_child(in, node.children[c]);
                                }
                            }
                            merge_group(in, soa.max_sh_degree, reps, ni);
                        }
                    });
            }
            const auto t_merge = elapsed_ms();
            if (progress && !progress(0.75f, "Moment matching")) {
                return std::unexpected("build_octree_lod: cancelled by user");
            }

            // BFS emission: single-splat leaves hoist their splat into the
            // parent's child block, every other octree node becomes one
            // interior output node. BFS order keeps children contiguous and
            // parents first, matching build_bhatt_lod's contract.
            size_t hoisted = 0;
            for (const OctNode& node : topo.nodes) {
                if (node.leaf && node.splat_count() == 1) {
                    ++hoisted;
                }
            }
            const bool root_is_single = topo.nodes[0].leaf && topo.nodes[0].splat_count() == 1;
            const size_t output_count = root_is_single ? 1 : n + node_count - hoisted;

            std::vector<uint32_t> source(output_count);
            std::vector<uint16_t> out_child_count(output_count, 0);
            std::vector<uint32_t> out_child_start(output_count, 0);
            std::vector<uint8_t> out_level(output_count, 0);
            {
                std::vector<std::pair<uint32_t, uint32_t>> queue; // (out index, octree node)
                queue.reserve(node_count - hoisted);
                if (root_is_single) {
                    source[0] = order[0];
                } else {
                    source[0] = kInteriorFlag | 0u;
                    queue.emplace_back(0u, 0u);
                }
                size_t next = 1;
                for (size_t head = 0; head < queue.size(); ++head) {
                    const auto [oi, ni] = queue[head];
                    const OctNode& node = topo.nodes[ni];
                    const auto child_level =
                        static_cast<uint8_t>(std::min<uint32_t>(out_level[oi] + 1u, 255u));
                    out_child_start[oi] = static_cast<uint32_t>(next);
                    if (node.leaf) {
                        out_child_count[oi] = static_cast<uint16_t>(node.splat_count());
                        for (uint32_t i = node.begin; i < node.end; ++i, ++next) {
                            source[next] = order[i];
                            out_level[next] = child_level;
                        }
                    } else {
                        out_child_count[oi] = node.child_count;
                        for (uint8_t c = 0; c < node.child_count; ++c, ++next) {
                            const uint32_t ci = node.children[c];
                            const OctNode& child = topo.nodes[ci];
                            if (child.leaf && child.splat_count() == 1) {
                                source[next] = order[child.begin];
                            } else {
                                source[next] = kInteriorFlag | ci;
                                queue.emplace_back(static_cast<uint32_t>(next),
                                                   static_cast<uint32_t>(ci));
                            }
                            out_level[next] = child_level;
                        }
                    }
                }
                assert(next == output_count);
            }

            if (options.leaf_input_indices != nullptr) {
                auto& leaf_map = *options.leaf_input_indices;
                leaf_map.assign(output_count, std::numeric_limits<uint32_t>::max());
                for (size_t i = 0; i < output_count; ++i) {
                    if ((source[i] & kInteriorFlag) == 0) {
                        leaf_map[i] = source[i];
                    }
                }
            }

            // Output tensors + tree metadata, mirroring build_bhatt_lod.
            const int max_sh = soa.max_sh_degree;
            const int shN_coeffs = static_cast<int>(input.max_sh_coeffs_rest());
            std::vector<float> means_vec(output_count * 3);
            std::vector<float> opacity_vec(output_count);
            std::vector<float> sh0_vec(output_count * 3);
            std::vector<float> scaling_vec(output_count * 3);
            std::vector<float> rotation_vec(output_count * 4);
            std::vector<float> shN_vec;
            if (shN_coeffs > 0) {
                shN_vec.resize(output_count * static_cast<size_t>(shN_coeffs) * 3);
            }
            auto lod_tree = std::make_unique<SplatLodTree>();
            lod_tree->centers.resize(output_count);
            lod_tree->sizes.resize(output_count);

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, output_count),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const bool interior = (source[i] & kInteriorFlag) != 0;
                        const size_t s = source[i] & ~kInteriorFlag;
                        const float* const mean = (interior ? reps.mean : soa.mean).data() + s * 3;
                        const float* const scale = (interior ? reps.scale : soa.scale).data() + s * 3;
                        const float* const quat = (interior ? reps.quat : soa.quat).data() + s * 4;
                        const float* const rgb = (interior ? reps.rgb : soa.rgb).data() + s * 3;
                        const float alpha = interior ? reps.alpha[s] : soa.alpha[s];

                        means_vec[i * 3 + 0] = mean[0];
                        means_vec[i * 3 + 1] = mean[1];
                        means_vec[i * 3 + 2] = mean[2];
                        opacity_vec[i] = std::max(alpha, 0.0f);
                        sh0_vec[i * 3 + 0] = (rgb[0] - 0.5f) / SH_C0;
                        sh0_vec[i * 3 + 1] = (rgb[1] - 0.5f) / SH_C0;
                        sh0_vec[i * 3 + 2] = (rgb[2] - 0.5f) / SH_C0;
                        scaling_vec[i * 3 + 0] = std::log(std::max(scale[0], 1e-8f));
                        scaling_vec[i * 3 + 1] = std::log(std::max(scale[1], 1e-8f));
                        scaling_vec[i * 3 + 2] = std::log(std::max(scale[2], 1e-8f));
                        std::copy_n(quat, 4, rotation_vec.data() + i * 4);

                        if (shN_coeffs > 0) {
                            float* const dst = shN_vec.data() + i * static_cast<size_t>(shN_coeffs) * 3;
                            if (max_sh >= 1 && shN_coeffs >= 3) {
                                std::copy_n((interior ? reps.sh1 : soa.sh1).data() + s * 9, 9, dst);
                            }
                            if (max_sh >= 2 && shN_coeffs >= 8) {
                                std::copy_n((interior ? reps.sh2 : soa.sh2).data() + s * 15, 15, dst + 9);
                            }
                            if (max_sh >= 3 && shN_coeffs >= 15) {
                                std::copy_n((interior ? reps.sh3 : soa.sh3).data() + s * 21, 21, dst + 24);
                            }
                        }

                        lod_tree->centers[i] = {mean[0], mean[1], mean[2]};
                        const float max_scale = std::max({scale[0], scale[1], scale[2]});
                        float expansion = 1.0f;
                        if (alpha > 1.0f) {
                            const float spark_lod_opacity = std::min(alpha * 4.0f - 3.0f, 5.0f);
                            expansion = 1.0f + 0.7f * (spark_lod_opacity - 1.0f);
                        }
                        lod_tree->sizes[i] = 2.0f * expansion * max_scale;
                    }
                });

            Tensor shN_tensor;
            if (shN_coeffs > 0) {
                shN_tensor = Tensor::from_vector(
                    shN_vec, {output_count, static_cast<size_t>(shN_coeffs), 3}, Device::CPU);
            }
            auto result = std::make_unique<SplatData>(
                max_sh,
                Tensor::from_vector(means_vec, {output_count, 3}, Device::CPU),
                Tensor::from_vector(sh0_vec, {output_count, 1, 3}, Device::CPU),
                std::move(shN_tensor),
                Tensor::from_vector(scaling_vec, {output_count, 3}, Device::CPU),
                Tensor::from_vector(rotation_vec, {output_count, 4}, Device::CPU),
                Tensor::from_vector(opacity_vec, {output_count, 1}, Device::CPU),
                1.0f);

            lod_tree->child_count = std::move(out_child_count);
            lod_tree->child_start = std::move(out_child_start);
            lod_tree->lod_level = std::move(out_level);
            const size_t chunk_count =
                (output_count + SplatLodTree::kChunkSplats - 1) / SplatLodTree::kChunkSplats;
            lod_tree->chunk_to_page.resize(chunk_count);
            lod_tree->page_to_chunk.resize(chunk_count);
            std::iota(lod_tree->chunk_to_page.begin(), lod_tree->chunk_to_page.end(), 0u);
            std::iota(lod_tree->page_to_chunk.begin(), lod_tree->page_to_chunk.end(), 0u);
            lod_tree->lod_opacity_encoded = true;
            result->lod_tree = std::move(lod_tree);

            LOG_DEBUG("build_octree_lod: {} splats -> {} nodes ({} octree, {} hoisted) in {} ms "
                      "(extract {}, sort {}, topology {}, merge {})",
                      n, output_count, node_count, hoisted, elapsed_ms(),
                      t_extract, t_sort - t_extract, t_topo - t_sort, t_merge - t_topo);

            if (progress && !progress(1.0f, "LOD tree complete")) {
                return std::unexpected("build_octree_lod: cancelled by user");
            }
            return result;
        } catch (const std::exception& e) {
            LOG_ERROR("build_octree_lod failed: {}", e.what());
            return std::unexpected(e.what());
        }
    }

} // namespace lfs::core
