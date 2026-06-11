/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/ply_to_rad_lod.hpp"
#include "core/bhatt_lod.hpp"
#include "core/logger.hpp"
#include "core/mapped_file.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "formats/rad.hpp"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {

        constexpr float SH_C0 = 0.28209479177387814f;
        // Pack-domain node payload: means3, rgb3, alpha, scales3, rot4.
        constexpr std::size_t kBaseNodeFloats = 14;
        constexpr std::size_t kChunkSplats = lfs::core::SplatLodTree::kChunkSplats;
        constexpr int kCellBitsPerAxis = 8;
        constexpr std::size_t kCellsPerAxis = std::size_t{1} << kCellBitsPerAxis;
        constexpr std::size_t kCellCount = kCellsPerAxis * kCellsPerAxis * kCellsPerAxis;

        using lfs::core::MappedFile;

        // ====================================================================
        // PLY header
        // ====================================================================

        struct PlyGaussianLayout {
            std::size_t vertex_count = 0;
            std::size_t stride = 0;
            std::size_t data_offset = 0;
            std::uint32_t pos[3] = {};
            std::uint32_t dc[3] = {};
            std::uint32_t opacity = 0;
            std::uint32_t scale[3] = {};
            std::uint32_t rot[4] = {};
            std::vector<std::uint32_t> rest; // byte offsets, PLY (channel-major) order
            int sh_degree = 0;
            int rest_coeffs = 0;
        };

        std::size_t ply_type_size(const std::string_view type) {
            if (type == "float" || type == "float32" || type == "int" || type == "int32" ||
                type == "uint" || type == "uint32") {
                return 4;
            }
            if (type == "double" || type == "float64" || type == "int64" || type == "uint64") {
                return 8;
            }
            if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") {
                return 2;
            }
            if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") {
                return 1;
            }
            return 0;
        }

        std::expected<PlyGaussianLayout, std::string> parse_ply_gaussian_header(
            const std::uint8_t* data, const std::size_t size) {
            constexpr std::size_t kMaxHeader = 1 << 20;
            const std::string_view head(reinterpret_cast<const char*>(data),
                                        std::min(size, kMaxHeader));
            const std::size_t end_pos = head.find("end_header\n");
            if (!head.starts_with("ply") || end_pos == std::string_view::npos) {
                return std::unexpected("not a PLY file or header too large");
            }

            PlyGaussianLayout layout;
            layout.data_offset = end_pos + std::strlen("end_header\n");

            struct FloatProp {
                std::string name;
                std::uint32_t offset;
            };
            std::vector<FloatProp> float_props;
            bool in_vertex_element = false;
            bool seen_vertex_element = false;
            bool little_endian = false;
            std::size_t offset = 0;

            std::size_t line_start = 0;
            while (line_start < end_pos) {
                std::size_t line_end = head.find('\n', line_start);
                if (line_end == std::string_view::npos || line_end > end_pos) {
                    line_end = end_pos;
                }
                std::string_view line = head.substr(line_start, line_end - line_start);
                if (!line.empty() && line.back() == '\r') {
                    line.remove_suffix(1);
                }
                line_start = line_end + 1;

                if (line.starts_with("format ")) {
                    little_endian = line.find("binary_little_endian") != std::string_view::npos;
                } else if (line.starts_with("element ")) {
                    if (line.starts_with("element vertex ")) {
                        if (seen_vertex_element) {
                            return std::unexpected("multiple vertex elements are not supported");
                        }
                        in_vertex_element = true;
                        seen_vertex_element = true;
                        const std::string_view count_str = line.substr(std::strlen("element vertex "));
                        std::uint64_t count = 0;
                        const auto [ptr, ec] = std::from_chars(
                            count_str.data(), count_str.data() + count_str.size(), count);
                        if (ec != std::errc{} || count == 0) {
                            return std::unexpected("invalid vertex count in PLY header");
                        }
                        layout.vertex_count = count;
                    } else {
                        if (!seen_vertex_element) {
                            return std::unexpected(
                                "PLY elements before the vertex element are not supported");
                        }
                        in_vertex_element = false;
                    }
                } else if (line.starts_with("property ") && in_vertex_element) {
                    if (line.starts_with("property list")) {
                        return std::unexpected("list properties on vertices are not supported");
                    }
                    const std::size_t type_start = std::strlen("property ");
                    const std::size_t type_end = line.find(' ', type_start);
                    if (type_end == std::string_view::npos) {
                        return std::unexpected("malformed property line in PLY header");
                    }
                    const std::string_view type = line.substr(type_start, type_end - type_start);
                    const std::string_view name = line.substr(type_end + 1);
                    const std::size_t type_size = ply_type_size(type);
                    if (type_size == 0) {
                        return std::unexpected(std::format("unsupported property type '{}'", type));
                    }
                    if (type == "float" || type == "float32") {
                        float_props.push_back({std::string(name), static_cast<std::uint32_t>(offset)});
                    }
                    offset += type_size;
                }
            }

            if (!little_endian) {
                return std::unexpected("only binary_little_endian PLY files are supported");
            }
            if (!seen_vertex_element) {
                return std::unexpected("PLY file has no vertex element");
            }
            layout.stride = offset;

            const auto find_prop = [&](const std::string_view name) -> std::optional<std::uint32_t> {
                for (const auto& p : float_props) {
                    if (p.name == name) {
                        return p.offset;
                    }
                }
                return std::nullopt;
            };

            const auto require_prop = [&](const std::string_view name,
                                          std::uint32_t& out) -> std::optional<std::string> {
                const auto found = find_prop(name);
                if (!found) {
                    return std::format("missing required float property '{}'", name);
                }
                out = *found;
                return std::nullopt;
            };

            const char* const pos_names[3] = {"x", "y", "z"};
            const char* const dc_names[3] = {"f_dc_0", "f_dc_1", "f_dc_2"};
            const char* const scale_names[3] = {"scale_0", "scale_1", "scale_2"};
            const char* const rot_names[4] = {"rot_0", "rot_1", "rot_2", "rot_3"};
            for (int i = 0; i < 3; ++i) {
                if (auto err = require_prop(pos_names[i], layout.pos[i])) {
                    return std::unexpected(*err);
                }
                if (auto err = require_prop(dc_names[i], layout.dc[i])) {
                    return std::unexpected(*err);
                }
                if (auto err = require_prop(scale_names[i], layout.scale[i])) {
                    return std::unexpected(*err);
                }
            }
            for (int i = 0; i < 4; ++i) {
                if (auto err = require_prop(rot_names[i], layout.rot[i])) {
                    return std::unexpected(*err);
                }
            }
            if (auto err = require_prop("opacity", layout.opacity)) {
                return std::unexpected(*err);
            }

            for (std::size_t i = 0;; ++i) {
                const auto found = find_prop(std::format("f_rest_{}", i));
                if (!found) {
                    break;
                }
                layout.rest.push_back(*found);
            }
            switch (layout.rest.size()) {
            case 0:
                layout.sh_degree = 0;
                layout.rest_coeffs = 0;
                break;
            case 9:
                layout.sh_degree = 1;
                layout.rest_coeffs = 3;
                break;
            case 24:
                layout.sh_degree = 2;
                layout.rest_coeffs = 8;
                break;
            case 45:
                layout.sh_degree = 3;
                layout.rest_coeffs = 15;
                break;
            default:
                return std::unexpected(std::format(
                    "unsupported f_rest_* count {} (expected 0, 9, 24, or 45)", layout.rest.size()));
            }

            return layout;
        }

        inline float read_f32(const std::uint8_t* vertex, const std::uint32_t offset) {
            float v;
            std::memcpy(&v, vertex + offset, sizeof(float));
            return v;
        }

        // ====================================================================
        // Morton bucketing
        // ====================================================================

        inline std::uint32_t spread_bits_8(std::uint32_t v) {
            v &= 0xFFu;
            v = (v | (v << 8)) & 0x00F00Fu;
            v = (v | (v << 4)) & 0x0C30C3u;
            v = (v | (v << 2)) & 0x249249u;
            return v;
        }

        struct SceneBounds {
            float min[3] = {std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};
            float max[3] = {std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest()};
            float inv_extent[3] = {0.0f, 0.0f, 0.0f};

            void finalize() {
                for (int a = 0; a < 3; ++a) {
                    const float extent = max[a] - min[a];
                    inv_extent[a] = extent > 0.0f ? 1.0f / extent : 0.0f;
                }
            }
        };

        inline std::uint32_t cell_of_position(const SceneBounds& bounds,
                                              const float x, const float y, const float z) {
            const float p[3] = {x, y, z};
            std::uint32_t cell[3];
            for (int a = 0; a < 3; ++a) {
                const float norm = (p[a] - bounds.min[a]) * bounds.inv_extent[a];
                const auto q = static_cast<std::int64_t>(norm * static_cast<float>(kCellsPerAxis));
                cell[a] = static_cast<std::uint32_t>(
                    std::clamp<std::int64_t>(q, 0, static_cast<std::int64_t>(kCellsPerAxis) - 1));
            }
            return spread_bits_8(cell[0]) | (spread_bits_8(cell[1]) << 1) | (spread_bits_8(cell[2]) << 2);
        }

        // ====================================================================
        // Temp file helpers
        // ====================================================================

        struct TempDirGuard {
            std::filesystem::path path;

            ~TempDirGuard() {
                if (path.empty()) {
                    return;
                }
                std::error_code ec;
                std::filesystem::remove_all(path, ec);
                if (ec) {
                    LOG_WARN("ply_to_rad_lod: failed to remove temp dir {}: {}",
                             lfs::core::path_to_utf8(path), ec.message());
                }
            }
        };

        std::uint64_t current_pid() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
        }

        bool process_alive(const std::uint64_t pid) {
#ifdef _WIN32
            HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
            if (handle == nullptr) {
                return false;
            }
            DWORD exit_code = 0;
            const bool alive = GetExitCodeProcess(handle, &exit_code) && exit_code == STILL_ACTIVE;
            CloseHandle(handle);
            return alive;
#else
            return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
        }

        // The in-process TempDirGuard cannot run after a crash or kill, so each
        // conversion sweeps scratch dirs that earlier runs of the same output
        // left behind. The dir name carries the owner PID; a dead owner means
        // the scratch is abandoned. Legacy dirs without a PID predate this
        // scheme and are always stale.
        void remove_stale_temp_dirs(const std::filesystem::path& parent, const std::string& stem_utf8) {
            const std::string legacy_name = std::format(".{}.lodtmp", stem_utf8);
            const std::string pid_prefix = legacy_name + ".";

            std::error_code ec;
            std::filesystem::directory_iterator it(parent, ec);
            if (ec) {
                return;
            }
            for (const auto& entry : it) {
                if (!entry.is_directory(ec)) {
                    continue;
                }
                const std::string name = lfs::core::path_to_utf8(entry.path().filename());
                bool stale = name == legacy_name;
                if (!stale && name.starts_with(pid_prefix)) {
                    std::uint64_t pid = 0;
                    const char* const first = name.data() + pid_prefix.size();
                    const char* const last = name.data() + name.size();
                    const auto [end, errc] = std::from_chars(first, last, pid);
                    stale = errc == std::errc{} && end == last &&
                            pid != current_pid() && !process_alive(pid);
                }
                if (stale) {
                    LOG_INFO("ply_to_rad_lod: removing stale temp dir {}",
                             lfs::core::path_to_utf8(entry.path()));
                    std::filesystem::remove_all(entry.path(), ec);
                    if (ec) {
                        LOG_WARN("ply_to_rad_lod: failed to remove stale temp dir {}: {}",
                                 lfs::core::path_to_utf8(entry.path()), ec.message());
                    }
                }
            }
        }

        struct FileCloser {
            void operator()(std::FILE* f) const {
                if (f != nullptr) {
                    std::fclose(f);
                }
            }
        };
        using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

        [[nodiscard]] bool write_exact(std::FILE* f, const void* data, const std::size_t bytes) {
            return std::fwrite(data, 1, bytes, f) == bytes;
        }

        [[nodiscard]] bool read_exact(std::FILE* f, void* data, const std::size_t bytes) {
            return std::fread(data, 1, bytes, f) == bytes;
        }

        // ====================================================================
        // Per-bucket subtree build
        // ====================================================================

        // Pack-domain node arrays for one bucket subtree, root at index 0.
        struct BucketNodes {
            std::size_t count = 0;
            std::vector<float> means;
            std::vector<float> rgb;
            std::vector<float> alpha;
            std::vector<float> scales;
            std::vector<float> rotation;
            std::vector<float> shN;
            std::vector<std::uint16_t> child_count;
            std::vector<std::uint32_t> child_start;
        };

        struct BucketSummary {
            std::uint64_t node_count = 0;
            std::array<float, 14> root{}; // means3, rgb3, alpha, scales3, rot4
            std::vector<float> root_shN;
            std::uint16_t root_child_count = 0;
            std::uint32_t root_child_start = 0;
            // Local BFS level offsets: level l occupies [level_starts[l], level_starts[l+1]).
            std::vector<std::uint32_t> level_starts;
            // Depth of this bucket's root in the global tree (top-tree leaf level).
            std::uint32_t root_depth = 0;
        };

        // Relabel a children-contiguous subtree (root at 0, parents before
        // children) into BFS level order. A level-ordered file makes a coarse
        // LOD cut a contiguous prefix, which is what keeps the paged viewer's
        // chunk working set small. Returns per-level start offsets.
        std::vector<std::uint32_t> relabel_level_order(BucketNodes& nodes, const int rest_coeffs,
                                                       std::vector<std::uint32_t>* aux = nullptr) {
            const std::size_t n = nodes.count;
            std::vector<std::uint8_t> level(n, 0);
            std::uint8_t max_level = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint32_t cc = nodes.child_count[i];
                if (cc == 0) {
                    continue;
                }
                const auto child_level = static_cast<std::uint8_t>(
                    std::min<std::uint32_t>(level[i] + 1u, 255u));
                max_level = std::max(max_level, child_level);
                const std::uint32_t cs = nodes.child_start[i];
                for (std::uint32_t c = 0; c < cc; ++c) {
                    level[cs + c] = child_level;
                }
            }

            std::vector<std::uint32_t> level_starts(static_cast<std::size_t>(max_level) + 2, 0);
            for (std::size_t i = 0; i < n; ++i) {
                ++level_starts[level[i] + 1];
            }
            for (std::size_t l = 1; l < level_starts.size(); ++l) {
                level_starts[l] += level_starts[l - 1];
            }

            // Stable counting sort by level keeps each parent's child block
            // contiguous in the new order.
            std::vector<std::uint32_t> new_index(n);
            {
                std::vector<std::uint32_t> cursor(level_starts.begin(), level_starts.end() - 1);
                for (std::size_t i = 0; i < n; ++i) {
                    new_index[i] = cursor[level[i]]++;
                }
            }

            BucketNodes out;
            out.count = n;
            out.means.resize(n * 3);
            out.rgb.resize(n * 3);
            out.alpha.resize(n);
            out.scales.resize(n * 3);
            out.rotation.resize(n * 4);
            out.shN.resize(nodes.shN.size());
            out.child_count.resize(n);
            out.child_start.resize(n);
            const std::size_t sh_floats = static_cast<std::size_t>(rest_coeffs) * 3;
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, n),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i != range.end(); ++i) {
                        const std::size_t j = new_index[i];
                        std::memcpy(out.means.data() + j * 3, nodes.means.data() + i * 3, 3 * sizeof(float));
                        std::memcpy(out.rgb.data() + j * 3, nodes.rgb.data() + i * 3, 3 * sizeof(float));
                        out.alpha[j] = nodes.alpha[i];
                        std::memcpy(out.scales.data() + j * 3, nodes.scales.data() + i * 3, 3 * sizeof(float));
                        std::memcpy(out.rotation.data() + j * 4, nodes.rotation.data() + i * 4, 4 * sizeof(float));
                        if (sh_floats > 0) {
                            std::memcpy(out.shN.data() + j * sh_floats,
                                        nodes.shN.data() + i * sh_floats,
                                        sh_floats * sizeof(float));
                        }
                        out.child_count[j] = nodes.child_count[i];
                        out.child_start[j] = nodes.child_count[i] > 0
                                                 ? new_index[nodes.child_start[i]]
                                                 : 0;
                    }
                });
            if (aux != nullptr) {
                std::vector<std::uint32_t> aux_out(n);
                for (std::size_t i = 0; i < n; ++i) {
                    aux_out[new_index[i]] = (*aux)[i];
                }
                *aux = std::move(aux_out);
            }
            nodes = std::move(out);
            return level_starts;
        }

        BucketNodes pack_lod_output(const SplatData& lod, const int rest_coeffs) {
            const std::size_t n = static_cast<std::size_t>(lod.size());
            BucketNodes nodes;
            nodes.count = n;
            nodes.means.resize(n * 3);
            nodes.rgb.resize(n * 3);
            nodes.alpha.resize(n);
            nodes.scales.resize(n * 3);
            nodes.rotation.resize(n * 4);
            if (rest_coeffs > 0) {
                nodes.shN.resize(n * static_cast<std::size_t>(rest_coeffs) * 3);
            }

            const auto means_cpu = lod.means_raw().cpu().contiguous();
            const auto sh0_cpu = lod.sh0_raw().cpu().contiguous();
            const auto opacity_cpu = lod.opacity_raw().cpu().contiguous();
            const auto scaling_cpu = lod.scaling_raw().cpu().contiguous();
            const auto rotation_cpu = lod.rotation_raw().cpu().contiguous();
            const float* const means_ptr = means_cpu.ptr<float>();
            const float* const sh0_ptr = sh0_cpu.ptr<float>();
            const float* const opacity_ptr = opacity_cpu.ptr<float>();
            const float* const scaling_ptr = scaling_cpu.ptr<float>();
            const float* const rotation_ptr = rotation_cpu.ptr<float>();

            Tensor shN_canon;
            const float* shN_ptr = nullptr;
            if (rest_coeffs > 0) {
                shN_canon = lod.shN_canonical_cpu();
                shN_ptr = shN_canon.ptr<float>();
            }

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, n),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i != range.end(); ++i) {
                        nodes.means[i * 3 + 0] = means_ptr[i * 3 + 0];
                        nodes.means[i * 3 + 1] = means_ptr[i * 3 + 1];
                        nodes.means[i * 3 + 2] = means_ptr[i * 3 + 2];
                        nodes.rgb[i * 3 + 0] = 0.5f + SH_C0 * sh0_ptr[i * 3 + 0];
                        nodes.rgb[i * 3 + 1] = 0.5f + SH_C0 * sh0_ptr[i * 3 + 1];
                        nodes.rgb[i * 3 + 2] = 0.5f + SH_C0 * sh0_ptr[i * 3 + 2];
                        nodes.alpha[i] = std::max(opacity_ptr[i], 0.0f);
                        nodes.scales[i * 3 + 0] = std::exp(scaling_ptr[i * 3 + 0]);
                        nodes.scales[i * 3 + 1] = std::exp(scaling_ptr[i * 3 + 1]);
                        nodes.scales[i * 3 + 2] = std::exp(scaling_ptr[i * 3 + 2]);
                        nodes.rotation[i * 4 + 0] = rotation_ptr[i * 4 + 0];
                        nodes.rotation[i * 4 + 1] = rotation_ptr[i * 4 + 1];
                        nodes.rotation[i * 4 + 2] = rotation_ptr[i * 4 + 2];
                        nodes.rotation[i * 4 + 3] = rotation_ptr[i * 4 + 3];
                        if (shN_ptr != nullptr) {
                            const std::size_t floats = static_cast<std::size_t>(rest_coeffs) * 3;
                            std::memcpy(nodes.shN.data() + i * floats,
                                        shN_ptr + i * floats,
                                        floats * sizeof(float));
                        }
                    }
                });

            const auto& tree = *lod.lod_tree;
            nodes.child_count.assign(tree.child_count.begin(), tree.child_count.begin() + n);
            nodes.child_start.assign(tree.child_start.begin(), tree.child_start.begin() + n);
            return nodes;
        }

        std::size_t bucket_node_record_bytes(const int rest_coeffs) {
            return (kBaseNodeFloats + static_cast<std::size_t>(rest_coeffs) * 3) * sizeof(float) +
                   sizeof(std::uint16_t) + sizeof(std::uint32_t);
        }

        [[nodiscard]] bool write_bucket_nodes(const std::filesystem::path& path, const BucketNodes& nodes,
                                              const int rest_coeffs) {
            FilePtr f(std::fopen(path.string().c_str(), "wb"));
            if (!f) {
                return false;
            }
            const std::uint64_t n = nodes.count;
            if (!write_exact(f.get(), &n, sizeof(n))) {
                return false;
            }
            const std::size_t record_bytes = bucket_node_record_bytes(rest_coeffs);
            const std::size_t sh_floats = static_cast<std::size_t>(rest_coeffs) * 3;
            constexpr std::size_t kBatch = 65536;
            std::vector<std::uint8_t> buffer(kBatch * record_bytes);
            for (std::size_t first = 0; first < nodes.count; first += kBatch) {
                const std::size_t batch = std::min(kBatch, nodes.count - first);
                for (std::size_t k = 0; k < batch; ++k) {
                    const std::size_t i = first + k;
                    std::uint8_t* rec = buffer.data() + k * record_bytes;
                    std::memcpy(rec, nodes.means.data() + i * 3, 3 * sizeof(float));
                    std::memcpy(rec + 12, nodes.rgb.data() + i * 3, 3 * sizeof(float));
                    std::memcpy(rec + 24, &nodes.alpha[i], sizeof(float));
                    std::memcpy(rec + 28, nodes.scales.data() + i * 3, 3 * sizeof(float));
                    std::memcpy(rec + 40, nodes.rotation.data() + i * 4, 4 * sizeof(float));
                    std::size_t off = 56;
                    if (sh_floats > 0) {
                        std::memcpy(rec + off, nodes.shN.data() + i * sh_floats, sh_floats * sizeof(float));
                        off += sh_floats * sizeof(float);
                    }
                    std::memcpy(rec + off, &nodes.child_count[i], sizeof(std::uint16_t));
                    std::memcpy(rec + off + 2, &nodes.child_start[i], sizeof(std::uint32_t));
                }
                if (!write_exact(f.get(), buffer.data(), batch * record_bytes)) {
                    return false;
                }
            }
            return true;
        }

        // Treelet layout (Phase C): global order stays band-major (a coarse
        // cut is still a file prefix), but inside a band each subtree's nodes
        // are grouped per band-root treelet so a chunk holds spatially local,
        // depth-adjacent nodes instead of a level-major slice of everything.
        constexpr std::size_t kTreeletBandLevels = 4;

        // Pass 1 input: one sequential batched scan of a bucket nodes file
        // extracting only the link tail of each fixed-size record.
        [[nodiscard]] bool read_bucket_links(const std::filesystem::path& path,
                                             const std::uint64_t expected_nodes,
                                             const int rest_coeffs,
                                             const bool skip_root,
                                             std::vector<std::uint16_t>& child_count,
                                             std::vector<std::uint32_t>& child_start) {
            FilePtr f(std::fopen(path.string().c_str(), "rb"));
            std::uint64_t n = 0;
            if (!f || !read_exact(f.get(), &n, sizeof(n)) || n != expected_nodes) {
                return false;
            }
            const std::size_t record_bytes = bucket_node_record_bytes(rest_coeffs);
            const std::size_t link_offset = record_bytes - sizeof(std::uint16_t) - sizeof(std::uint32_t);
            std::size_t first = 0;
            std::size_t total = static_cast<std::size_t>(expected_nodes);
            if (skip_root) {
                if (std::fseek(f.get(), static_cast<long>(record_bytes), SEEK_CUR) != 0) {
                    return false;
                }
                total -= 1;
            }
            child_count.resize(total);
            child_start.resize(total);
            constexpr std::size_t kBatch = 65536;
            std::vector<std::uint8_t> buffer(kBatch * record_bytes);
            while (first < total) {
                const std::size_t take = std::min(kBatch, total - first);
                if (!read_exact(f.get(), buffer.data(), take * record_bytes)) {
                    return false;
                }
                for (std::size_t i = 0; i < take; ++i) {
                    const std::uint8_t* const rec = buffer.data() + i * record_bytes + link_offset;
                    std::memcpy(&child_count[first + i], rec, sizeof(std::uint16_t));
                    std::memcpy(&child_start[first + i], rec + sizeof(std::uint16_t), sizeof(std::uint32_t));
                }
                first += take;
            }
            return true;
        }

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
        [[nodiscard]] TreeletRanks build_treelet_ranks(
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

        [[nodiscard]] bool read_node_slice(std::FILE* f, const std::size_t count, const int rest_coeffs,
                                           BucketNodes& nodes) {
            const std::size_t record_bytes = bucket_node_record_bytes(rest_coeffs);
            const std::size_t sh_floats = static_cast<std::size_t>(rest_coeffs) * 3;
            std::vector<std::uint8_t> buffer(count * record_bytes);
            if (!read_exact(f, buffer.data(), buffer.size())) {
                return false;
            }
            nodes.count = count;
            nodes.means.resize(count * 3);
            nodes.rgb.resize(count * 3);
            nodes.alpha.resize(count);
            nodes.scales.resize(count * 3);
            nodes.rotation.resize(count * 4);
            nodes.shN.resize(sh_floats * count);
            nodes.child_count.resize(count);
            nodes.child_start.resize(count);
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, count),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i != range.end(); ++i) {
                        const std::uint8_t* rec = buffer.data() + i * record_bytes;
                        std::memcpy(nodes.means.data() + i * 3, rec, 3 * sizeof(float));
                        std::memcpy(nodes.rgb.data() + i * 3, rec + 12, 3 * sizeof(float));
                        std::memcpy(&nodes.alpha[i], rec + 24, sizeof(float));
                        std::memcpy(nodes.scales.data() + i * 3, rec + 28, 3 * sizeof(float));
                        std::memcpy(nodes.rotation.data() + i * 4, rec + 40, 4 * sizeof(float));
                        std::size_t off = 56;
                        if (sh_floats > 0) {
                            std::memcpy(nodes.shN.data() + i * sh_floats, rec + off, sh_floats * sizeof(float));
                            off += sh_floats * sizeof(float);
                        }
                        std::memcpy(&nodes.child_count[i], rec + off, sizeof(std::uint16_t));
                        std::memcpy(&nodes.child_start[i], rec + off + 2, sizeof(std::uint32_t));
                    }
                });
            return true;
        }

        // Raw-domain record as written by the scatter pass.
        // [x y z dc0 dc1 dc2 opacity s0 s1 s2 r0 r1 r2 r3] + rest (canonical).
        constexpr std::size_t kBaseRecordFloats = 14;

        SplatData splat_data_from_records(const std::vector<float>& records,
                                          const std::size_t count,
                                          const int sh_degree,
                                          const int rest_coeffs) {
            const std::size_t record_floats = kBaseRecordFloats + static_cast<std::size_t>(rest_coeffs) * 3;
            std::vector<float> means(count * 3);
            std::vector<float> sh0(count * 3);
            std::vector<float> opacity(count);
            std::vector<float> scaling(count * 3);
            std::vector<float> rotation(count * 4);
            std::vector<float> shN(rest_coeffs > 0 ? count * static_cast<std::size_t>(rest_coeffs) * 3 : 0);

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, count),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i != range.end(); ++i) {
                        const float* const rec = records.data() + i * record_floats;
                        means[i * 3 + 0] = rec[0];
                        means[i * 3 + 1] = rec[1];
                        means[i * 3 + 2] = rec[2];
                        sh0[i * 3 + 0] = rec[3];
                        sh0[i * 3 + 1] = rec[4];
                        sh0[i * 3 + 2] = rec[5];
                        opacity[i] = rec[6];
                        scaling[i * 3 + 0] = rec[7];
                        scaling[i * 3 + 1] = rec[8];
                        scaling[i * 3 + 2] = rec[9];
                        rotation[i * 4 + 0] = rec[10];
                        rotation[i * 4 + 1] = rec[11];
                        rotation[i * 4 + 2] = rec[12];
                        rotation[i * 4 + 3] = rec[13];
                        if (rest_coeffs > 0) {
                            std::memcpy(shN.data() + i * static_cast<std::size_t>(rest_coeffs) * 3,
                                        rec + kBaseRecordFloats,
                                        static_cast<std::size_t>(rest_coeffs) * 3 * sizeof(float));
                        }
                    }
                });

            Tensor shN_tensor;
            if (rest_coeffs > 0) {
                shN_tensor = Tensor::from_vector(
                    shN, {count, static_cast<std::size_t>(rest_coeffs), 3}, Device::CPU);
            }
            return SplatData(
                sh_degree,
                Tensor::from_vector(means, {count, 3}, Device::CPU),
                Tensor::from_vector(sh0, {count, 1, 3}, Device::CPU),
                std::move(shN_tensor),
                Tensor::from_vector(scaling, {count, 3}, Device::CPU),
                Tensor::from_vector(rotation, {count, 4}, Device::CPU),
                Tensor::from_vector(opacity, {count, 1}, Device::CPU),
                1.0f);
        }

        // Single-splat bucket: the splat itself becomes the subtree root.
        BucketNodes pack_single_record(const float* rec, const int rest_coeffs) {
            BucketNodes nodes;
            nodes.count = 1;
            nodes.means = {rec[0], rec[1], rec[2]};
            nodes.rgb = {0.5f + SH_C0 * rec[3], 0.5f + SH_C0 * rec[4], 0.5f + SH_C0 * rec[5]};
            nodes.alpha = {1.0f / (1.0f + std::exp(-rec[6]))};
            nodes.scales = {std::exp(rec[7]), std::exp(rec[8]), std::exp(rec[9])};
            const float qw = rec[10], qx = rec[11], qy = rec[12], qz = rec[13];
            const float inv_norm =
                1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), 1e-12f);
            nodes.rotation = {qw * inv_norm, qx * inv_norm, qy * inv_norm, qz * inv_norm};
            if (rest_coeffs > 0) {
                nodes.shN.assign(rec + kBaseRecordFloats,
                                 rec + kBaseRecordFloats + static_cast<std::size_t>(rest_coeffs) * 3);
            }
            nodes.child_count = {0};
            nodes.child_start = {0};
            return nodes;
        }

        BucketSummary summarize_bucket(const BucketNodes& nodes, const int rest_coeffs) {
            BucketSummary summary;
            summary.node_count = nodes.count;
            summary.root = {nodes.means[0], nodes.means[1], nodes.means[2],
                            nodes.rgb[0], nodes.rgb[1], nodes.rgb[2],
                            nodes.alpha[0],
                            nodes.scales[0], nodes.scales[1], nodes.scales[2],
                            nodes.rotation[0], nodes.rotation[1], nodes.rotation[2], nodes.rotation[3]};
            if (rest_coeffs > 0) {
                summary.root_shN.assign(nodes.shN.begin(),
                                        nodes.shN.begin() + static_cast<std::size_t>(rest_coeffs) * 3);
            }
            summary.root_child_count = nodes.child_count[0];
            summary.root_child_start = nodes.child_start[0];
            return summary;
        }

        // ====================================================================
        // Streaming chunk accumulator
        // ====================================================================

        // Stages whole chunks and hands them to the writer in batches so the
        // per-chunk DEFLATE runs in parallel instead of serializing the final
        // assembly phase. Staging is bounded (~256 MB) and reused across
        // batches.
        class ChunkAccumulator {
        public:
            ChunkAccumulator(RadStreamWriter& writer, const int rest_coeffs)
                : writer_(writer),
                  rest_coeffs_(rest_coeffs) {
                const std::size_t chunk_bytes =
                    kChunkSplats *
                    ((14 + static_cast<std::size_t>(rest_coeffs_) * 3) * sizeof(float) + 6);
                const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
                batch_capacity_ = std::clamp<std::size_t>(
                    (std::size_t{256} << 20) / chunk_bytes, 4, std::max<std::size_t>(hw * 2, 4));
            }

            [[nodiscard]] std::expected<void, std::string> push_span(
                const BucketNodes& nodes, std::size_t first, std::size_t count) {
                while (count > 0) {
                    Stage& stage = filling_stage();
                    const std::size_t take = std::min(kChunkSplats - stage.fill, count);
                    copy_range(stage, nodes, first, take);
                    stage.fill += take;
                    first += take;
                    count -= take;
                    if (stage.fill == kChunkSplats) {
                        ++full_stages_;
                        if (full_stages_ == batch_capacity_) {
                            if (auto ok = flush_batch(); !ok) {
                                return ok;
                            }
                        }
                    }
                }
                return {};
            }

            [[nodiscard]] std::expected<void, std::string> finish() {
                return flush_batch();
            }

        private:
            struct Stage {
                std::vector<float> means, rgb, alpha, scales, rotation, shN;
                std::vector<std::uint16_t> child_count;
                std::vector<std::uint32_t> child_start;
                std::size_t fill = 0;
            };

            Stage& filling_stage() {
                if (full_stages_ == stages_.size()) {
                    Stage stage;
                    stage.means.resize(kChunkSplats * 3);
                    stage.rgb.resize(kChunkSplats * 3);
                    stage.alpha.resize(kChunkSplats);
                    stage.scales.resize(kChunkSplats * 3);
                    stage.rotation.resize(kChunkSplats * 4);
                    if (rest_coeffs_ > 0) {
                        stage.shN.resize(kChunkSplats * static_cast<std::size_t>(rest_coeffs_) * 3);
                    }
                    stage.child_count.resize(kChunkSplats);
                    stage.child_start.resize(kChunkSplats);
                    stages_.push_back(std::move(stage));
                }
                return stages_[full_stages_];
            }

            void copy_range(Stage& stage, const BucketNodes& nodes,
                            const std::size_t first, const std::size_t take) {
                const std::size_t fill = stage.fill;
                std::memcpy(stage.means.data() + fill * 3, nodes.means.data() + first * 3,
                            take * 3 * sizeof(float));
                std::memcpy(stage.rgb.data() + fill * 3, nodes.rgb.data() + first * 3,
                            take * 3 * sizeof(float));
                std::memcpy(stage.alpha.data() + fill, nodes.alpha.data() + first,
                            take * sizeof(float));
                std::memcpy(stage.scales.data() + fill * 3, nodes.scales.data() + first * 3,
                            take * 3 * sizeof(float));
                std::memcpy(stage.rotation.data() + fill * 4, nodes.rotation.data() + first * 4,
                            take * 4 * sizeof(float));
                if (rest_coeffs_ > 0) {
                    const std::size_t floats = static_cast<std::size_t>(rest_coeffs_) * 3;
                    std::memcpy(stage.shN.data() + fill * floats, nodes.shN.data() + first * floats,
                                take * floats * sizeof(float));
                }
                std::memcpy(stage.child_count.data() + fill, nodes.child_count.data() + first,
                            take * sizeof(std::uint16_t));
                std::memcpy(stage.child_start.data() + fill, nodes.child_start.data() + first,
                            take * sizeof(std::uint32_t));
            }

            [[nodiscard]] std::expected<void, std::string> flush_batch() {
                std::size_t pending = full_stages_;
                if (pending < stages_.size() && stages_[pending].fill > 0) {
                    ++pending; // trailing partial chunk, only legal as the file's last
                }
                if (pending == 0) {
                    return {};
                }
                std::vector<RadStreamChunkSource> sources(pending);
                for (std::size_t i = 0; i < pending; ++i) {
                    const Stage& stage = stages_[i];
                    auto& chunk = sources[i];
                    chunk.count = static_cast<std::uint32_t>(stage.fill);
                    chunk.means = stage.means.data();
                    chunk.rgb = stage.rgb.data();
                    chunk.alpha = stage.alpha.data();
                    chunk.scales = stage.scales.data();
                    chunk.rotation = stage.rotation.data();
                    chunk.shN = rest_coeffs_ > 0 ? stage.shN.data() : nullptr;
                    chunk.child_count = stage.child_count.data();
                    chunk.child_start = stage.child_start.data();
                }
                auto ok = writer_.append_batch(sources);
                for (std::size_t i = 0; i < pending; ++i) {
                    stages_[i].fill = 0;
                }
                full_stages_ = 0;
                return ok;
            }

            RadStreamWriter& writer_;
            int rest_coeffs_;
            std::size_t batch_capacity_ = 0;
            std::size_t full_stages_ = 0;
            std::vector<Stage> stages_;
        };

        std::size_t available_memory_bytes() {
#ifdef _WIN32
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            if (GlobalMemoryStatusEx(&status)) {
                return static_cast<std::size_t>(status.ullAvailPhys);
            }
#else
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.starts_with("MemAvailable:")) {
                    std::uint64_t kb = 0;
                    std::sscanf(line.c_str(), "MemAvailable: %lu kB", &kb);
                    return static_cast<std::size_t>(kb) * 1024;
                }
            }
#endif
            return std::size_t{8} * 1024 * 1024 * 1024;
        }

    } // namespace

    std::expected<PlyGaussianInfo, std::string> probe_ply_gaussians(
        const std::filesystem::path& input_path) {
        MappedFile file;
        if (!file.open(input_path)) {
            return std::unexpected("failed to open PLY file");
        }
        auto layout = parse_ply_gaussian_header(file.data(), file.size());
        if (!layout) {
            return std::unexpected(layout.error());
        }
        return PlyGaussianInfo{
            .vertex_count = layout->vertex_count,
            .sh_degree = layout->sh_degree,
        };
    }

    Result<void> convert_ply_to_rad_lod(const std::filesystem::path& input_path,
                                        const std::filesystem::path& output_path,
                                        const PlyToRadLodOptions& options) {
        const auto t_start = std::chrono::high_resolution_clock::now();

        const auto report = [&](const float progress, const std::string& stage) -> bool {
            if (options.progress) {
                return options.progress(progress, stage);
            }
            return true;
        };
        const auto cancelled = [&]() -> Result<void> {
            return make_error(ErrorCode::CANCELLED, "PLY to RAD LOD conversion cancelled", input_path);
        };

        MappedFile file;
        if (!file.open(input_path)) {
            return make_error(ErrorCode::READ_FAILURE, "Failed to open PLY file", input_path);
        }

        auto layout_result = parse_ply_gaussian_header(file.data(), file.size());
        if (!layout_result) {
            return make_error(ErrorCode::INVALID_HEADER, layout_result.error(), input_path);
        }
        const PlyGaussianLayout layout = std::move(*layout_result);

        const std::size_t vertex_bytes = layout.vertex_count * layout.stride;
        if (layout.data_offset + vertex_bytes > file.size()) {
            return make_error(ErrorCode::CORRUPTED_DATA, "PLY vertex data exceeds file size", input_path);
        }
        const std::uint8_t* const vertex_base = file.data() + layout.data_offset;
        const std::size_t N = layout.vertex_count;
        const int rest_coeffs = layout.rest_coeffs;
        const std::size_t record_floats = kBaseRecordFloats + static_cast<std::size_t>(rest_coeffs) * 3;

        LOG_INFO("ply_to_rad_lod: {} splats, SH degree {}, stride {} bytes",
                 N, layout.sh_degree, layout.stride);

        // Scratch space lives next to the output by default (same drive). The
        // PID suffix keeps concurrent conversions of the same output from
        // deleting each other's bucket files mid-run.
        TempDirGuard temp_guard;
        {
            std::filesystem::path temp_root = options.temp_dir;
            if (temp_root.empty()) {
                const std::string stem_utf8 = lfs::core::path_to_utf8(output_path.stem());
                remove_stale_temp_dirs(output_path.parent_path(), stem_utf8);
                temp_root = output_path.parent_path() /
                            std::format(".{}.lodtmp.{}", stem_utf8, current_pid());
            }
            std::error_code ec;
            std::filesystem::create_directories(temp_root, ec);
            if (ec) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to create temp dir: {}", ec.message()),
                                  temp_root);
            }
            temp_guard.path = temp_root;
        }

        // ------------------------------------------------------------------
        // Pass A: scene bounds. Robust percentiles instead of min/max: sparse
        // scanner outliers otherwise stretch the grid until the dense region
        // collapses into a few unsplittable cells, and the largest bucket's
        // serial LOD build becomes the conversion's critical path. Positions
        // outside the robust box clamp into edge cells in cell_of_position.
        // ------------------------------------------------------------------
        if (!report(0.0f, "Scanning scene bounds")) {
            return cancelled();
        }

        SceneBounds bounds;
        {
            const std::size_t step = std::max<std::size_t>(N >> 20, 1);
            const std::size_t sample_count = (N + step - 1) / step;
            std::array<std::vector<float>, 3> samples;
            for (auto& axis_samples : samples) {
                axis_samples.resize(sample_count);
            }
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, sample_count, 1 << 14),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t k = range.begin(); k != range.end(); ++k) {
                        const std::uint8_t* const vertex = vertex_base + (k * step) * layout.stride;
                        for (int a = 0; a < 3; ++a) {
                            samples[a][k] = read_f32(vertex, layout.pos[a]);
                        }
                    }
                });
            for (int a = 0; a < 3; ++a) {
                auto& axis_samples = samples[a];
                axis_samples.erase(
                    std::remove_if(axis_samples.begin(), axis_samples.end(),
                                   [](const float v) { return !std::isfinite(v); }),
                    axis_samples.end());
                if (axis_samples.empty()) {
                    return make_error(ErrorCode::CORRUPTED_DATA,
                                      "PLY contains no finite positions", input_path);
                }
                const std::size_t tail = axis_samples.size() / 1000;
                const std::size_t lo = tail;
                const std::size_t hi = axis_samples.size() - 1 - tail;
                std::nth_element(axis_samples.begin(), axis_samples.begin() + lo, axis_samples.end());
                const float q_lo = axis_samples[lo];
                std::nth_element(axis_samples.begin() + lo, axis_samples.begin() + hi, axis_samples.end());
                const float q_hi = axis_samples[hi];
                const float margin = std::max((q_hi - q_lo) * 0.02f, 1e-6f);
                bounds.min[a] = q_lo - margin;
                bounds.max[a] = q_hi + margin;
            }
        }
        bounds.finalize();

        if (!report(0.03f, "Computing spatial histogram")) {
            return cancelled();
        }

        // ------------------------------------------------------------------
        // Pass B: Morton cell histogram + greedy bucket grouping
        // ------------------------------------------------------------------
        std::unique_ptr<std::atomic<std::uint32_t>[]> cell_hist(
            new std::atomic<std::uint32_t>[kCellCount]());
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, N, 1 << 18),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t i = range.begin(); i != range.end(); ++i) {
                    const std::uint8_t* const vertex = vertex_base + i * layout.stride;
                    const std::uint32_t cell = cell_of_position(bounds,
                                                                read_f32(vertex, layout.pos[0]),
                                                                read_f32(vertex, layout.pos[1]),
                                                                read_f32(vertex, layout.pos[2]));
                    cell_hist[cell].fetch_add(1, std::memory_order_relaxed);
                }
            });

        const std::size_t target_bucket = std::max<std::size_t>(options.target_bucket_splats, 65536);
        std::vector<std::uint32_t> cell_to_bucket(kCellCount, 0);
        std::vector<std::uint64_t> bucket_counts;
        {
            std::uint64_t acc = 0;
            std::uint32_t bucket = 0;
            for (std::size_t cell = 0; cell < kCellCount; ++cell) {
                const std::uint32_t c = cell_hist[cell].load(std::memory_order_relaxed);
                if (acc > 0 && acc + c > target_bucket) {
                    bucket_counts.push_back(acc);
                    ++bucket;
                    acc = 0;
                }
                cell_to_bucket[cell] = bucket;
                acc += c;
            }
            bucket_counts.push_back(acc);
            while (!bucket_counts.empty() && bucket_counts.back() == 0) {
                bucket_counts.pop_back();
            }
        }
        cell_hist.reset();

        const std::size_t bucket_count = bucket_counts.size();
        if (bucket_count == 0) {
            return make_error(ErrorCode::CORRUPTED_DATA, "PLY contains no splats", input_path);
        }
        for (std::size_t b = 0; b < bucket_count; ++b) {
            if (bucket_counts[b] > 4 * target_bucket) {
                LOG_WARN("ply_to_rad_lod: bucket {} holds {} splats (target {}); "
                         "a single dense Morton cell exceeds the bucket budget",
                         b, bucket_counts[b], target_bucket);
            }
        }
        LOG_INFO("ply_to_rad_lod: {} buckets (target {} splats each)", bucket_count, target_bucket);

        if (!report(0.06f, "Scattering splats into spatial buckets")) {
            return cancelled();
        }

        // ------------------------------------------------------------------
        // Pass C: scatter raw records into per-bucket files
        // ------------------------------------------------------------------
        const auto bucket_records_path = [&](const std::size_t b) {
            return temp_guard.path / std::format("bucket_{:05}.records", b);
        };
        const auto bucket_nodes_path = [&](const std::size_t b) {
            return temp_guard.path / std::format("bucket_{:05}.nodes", b);
        };

        {
            std::vector<FilePtr> bucket_files(bucket_count);
            std::vector<std::mutex> bucket_mutexes(bucket_count);
            for (std::size_t b = 0; b < bucket_count; ++b) {
                bucket_files[b].reset(std::fopen(bucket_records_path(b).string().c_str(), "wb"));
                if (!bucket_files[b]) {
                    return make_error(ErrorCode::WRITE_FAILURE,
                                      std::format("Failed to create bucket file: {}", std::strerror(errno)),
                                      bucket_records_path(b));
                }
            }

            constexpr std::size_t kFlushFloats = 64 * 1024; // 256 KB per bucket buffer
            struct ScatterState {
                std::vector<std::vector<float>> pending;
            };
            tbb::enumerable_thread_specific<ScatterState> states([&] {
                ScatterState s;
                s.pending.resize(bucket_count);
                return s;
            });

            std::atomic<int> write_errno{0};
            const auto flush_bucket = [&](const std::size_t b, std::vector<float>& buf) {
                if (buf.empty()) {
                    return;
                }
                std::lock_guard<std::mutex> lock(bucket_mutexes[b]);
                if (!write_exact(bucket_files[b].get(), buf.data(), buf.size() * sizeof(float))) {
                    int expected = 0;
                    write_errno.compare_exchange_strong(expected, errno != 0 ? errno : EIO);
                }
                buf.clear();
            };

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, N, 1 << 16),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    auto& state = states.local();
                    for (std::size_t i = range.begin(); i != range.end(); ++i) {
                        const std::uint8_t* const vertex = vertex_base + i * layout.stride;
                        const float x = read_f32(vertex, layout.pos[0]);
                        const float y = read_f32(vertex, layout.pos[1]);
                        const float z = read_f32(vertex, layout.pos[2]);
                        const std::uint32_t bucket = cell_to_bucket[cell_of_position(bounds, x, y, z)];
                        auto& buf = state.pending[bucket];

                        buf.push_back(x);
                        buf.push_back(y);
                        buf.push_back(z);
                        buf.push_back(read_f32(vertex, layout.dc[0]));
                        buf.push_back(read_f32(vertex, layout.dc[1]));
                        buf.push_back(read_f32(vertex, layout.dc[2]));
                        buf.push_back(read_f32(vertex, layout.opacity));
                        buf.push_back(read_f32(vertex, layout.scale[0]));
                        buf.push_back(read_f32(vertex, layout.scale[1]));
                        buf.push_back(read_f32(vertex, layout.scale[2]));
                        buf.push_back(read_f32(vertex, layout.rot[0]));
                        buf.push_back(read_f32(vertex, layout.rot[1]));
                        buf.push_back(read_f32(vertex, layout.rot[2]));
                        buf.push_back(read_f32(vertex, layout.rot[3]));
                        // f_rest is channel-major in the file; records store
                        // canonical [coeff][channel] order.
                        for (int coeff = 0; coeff < rest_coeffs; ++coeff) {
                            for (int ch = 0; ch < 3; ++ch) {
                                buf.push_back(read_f32(
                                    vertex, layout.rest[static_cast<std::size_t>(ch) * rest_coeffs + coeff]));
                            }
                        }

                        if (buf.size() >= kFlushFloats) {
                            flush_bucket(bucket, buf);
                        }
                    }
                });

            for (auto& state : states) {
                for (std::size_t b = 0; b < bucket_count; ++b) {
                    flush_bucket(b, state.pending[b]);
                }
            }
            if (const int err = write_errno.load(); err != 0) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to write bucket records: {}", std::strerror(err)),
                                  temp_guard.path);
            }
        }
        file.close();

        if (!report(0.18f, "Building per-bucket LOD subtrees")) {
            return cancelled();
        }

        // ------------------------------------------------------------------
        // Per-bucket LOD subtree builds (bounded concurrency)
        // ------------------------------------------------------------------
        std::size_t concurrent_buckets = options.max_concurrent_buckets;
        if (concurrent_buckets == 0) {
            // Workset (~150-420 B by SH degree, 1.5x capacity), raw records and
            // packed output all coexist per in-flight bucket. The merge loop is
            // mostly serial per bucket, so concurrency up to the core count is
            // what keeps the machine busy; the memory budget is the real bound.
            const std::size_t per_splat_estimate =
                record_floats * sizeof(float) + 240 + static_cast<std::size_t>(rest_coeffs) * 30;
            const std::size_t per_bucket_estimate = target_bucket * per_splat_estimate;
            const std::size_t budget = available_memory_bytes() / 2;
            const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
            concurrent_buckets = std::clamp<std::size_t>(
                per_bucket_estimate > 0 ? budget / per_bucket_estimate : 1, 1, hw);
        }
        LOG_INFO("ply_to_rad_lod: building {} bucket subtrees, {} in flight",
                 bucket_count, std::min(concurrent_buckets, bucket_count));

        std::vector<BucketSummary> summaries(bucket_count);
        {
            std::atomic<std::size_t> next_bucket{0};
            std::atomic<std::size_t> done_buckets{0};
            std::atomic<bool> failed{false};
            std::atomic<bool> cancel_requested{false};
            std::mutex error_mutex;
            std::string first_error;
            std::mutex progress_mutex;

            const auto fail = [&](std::string msg) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) {
                    first_error = std::move(msg);
                }
            };

            const auto worker = [&]() {
                while (true) {
                    const std::size_t b = next_bucket.fetch_add(1);
                    if (b >= bucket_count || failed.load(std::memory_order_relaxed) ||
                        cancel_requested.load(std::memory_order_relaxed)) {
                        return;
                    }

                    const std::size_t count = static_cast<std::size_t>(bucket_counts[b]);
                    std::vector<float> records(count * record_floats);
                    {
                        FilePtr f(std::fopen(bucket_records_path(b).string().c_str(), "rb"));
                        if (!f || !read_exact(f.get(), records.data(), records.size() * sizeof(float))) {
                            fail(std::format("failed to read bucket records {}: {}", b, std::strerror(errno)));
                            return;
                        }
                    }
                    std::error_code ec;
                    std::filesystem::remove(bucket_records_path(b), ec);

                    BucketNodes nodes;
                    if (count == 1) {
                        nodes = pack_single_record(records.data(), rest_coeffs);
                    } else {
                        SplatData bucket_input =
                            splat_data_from_records(records, count, layout.sh_degree, rest_coeffs);
                        records.clear();
                        records.shrink_to_fit();

                        lfs::core::BhattLodBuildOptions lod_options;
                        lod_options.lod_base = options.lod_base;
                        auto lod_result = lfs::core::build_bhatt_lod(bucket_input, lod_options);
                        if (!lod_result) {
                            fail(std::format("bucket {} LOD build failed: {}", b, lod_result.error()));
                            return;
                        }
                        nodes = pack_lod_output(**lod_result, rest_coeffs);
                    }

                    auto level_starts = relabel_level_order(nodes, rest_coeffs);
                    summaries[b] = summarize_bucket(nodes, rest_coeffs);
                    summaries[b].level_starts = std::move(level_starts);
                    if (!write_bucket_nodes(bucket_nodes_path(b), nodes, rest_coeffs)) {
                        fail(std::format("failed to write bucket nodes {}: {}", b, std::strerror(errno)));
                        return;
                    }

                    const std::size_t done = done_buckets.fetch_add(1) + 1;
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        const float p = 0.18f + 0.57f * static_cast<float>(done) /
                                                    static_cast<float>(bucket_count);
                        if (!report(p, std::format("LOD subtree {}/{}", done, bucket_count))) {
                            cancel_requested.store(true, std::memory_order_relaxed);
                        }
                    }
                }
            };

            std::vector<std::thread> threads;
            const std::size_t worker_count = std::min(concurrent_buckets, bucket_count);
            threads.reserve(worker_count);
            for (std::size_t t = 0; t < worker_count; ++t) {
                threads.emplace_back(worker);
            }
            for (auto& t : threads) {
                t.join();
            }

            if (cancel_requested.load()) {
                return cancelled();
            }
            if (failed.load()) {
                return make_error(ErrorCode::CORRUPTED_DATA, first_error, input_path);
            }
        }

        if (!report(0.76f, "Building top-level LOD tree")) {
            return cancelled();
        }

        // ------------------------------------------------------------------
        // Top tree over bucket roots
        // ------------------------------------------------------------------
        BucketNodes top_nodes;
        std::vector<std::uint32_t> top_leaf_bucket; // per top node: bucket id or UINT32_MAX
        std::vector<std::uint32_t> top_level_starts;
        if (bucket_count == 1) {
            // Single bucket: its subtree is the whole hierarchy.
            top_nodes.count = 0;
        } else {
            const std::size_t roots = bucket_count;
            std::vector<float> means(roots * 3);
            std::vector<float> sh0(roots * 3);
            std::vector<float> opacity(roots);
            std::vector<float> scaling(roots * 3);
            std::vector<float> rotation(roots * 4);
            std::vector<float> shN(rest_coeffs > 0 ? roots * static_cast<std::size_t>(rest_coeffs) * 3 : 0);
            for (std::size_t b = 0; b < roots; ++b) {
                const auto& r = summaries[b].root;
                means[b * 3 + 0] = r[0];
                means[b * 3 + 1] = r[1];
                means[b * 3 + 2] = r[2];
                sh0[b * 3 + 0] = (r[3] - 0.5f) / SH_C0;
                sh0[b * 3 + 1] = (r[4] - 0.5f) / SH_C0;
                sh0[b * 3 + 2] = (r[5] - 0.5f) / SH_C0;
                opacity[b] = r[6];
                scaling[b * 3 + 0] = std::log(std::max(r[7], 1e-8f));
                scaling[b * 3 + 1] = std::log(std::max(r[8], 1e-8f));
                scaling[b * 3 + 2] = std::log(std::max(r[9], 1e-8f));
                rotation[b * 4 + 0] = r[10];
                rotation[b * 4 + 1] = r[11];
                rotation[b * 4 + 2] = r[12];
                rotation[b * 4 + 3] = r[13];
                if (rest_coeffs > 0) {
                    std::memcpy(shN.data() + b * static_cast<std::size_t>(rest_coeffs) * 3,
                                summaries[b].root_shN.data(),
                                static_cast<std::size_t>(rest_coeffs) * 3 * sizeof(float));
                }
            }

            Tensor shN_tensor;
            if (rest_coeffs > 0) {
                shN_tensor = Tensor::from_vector(
                    shN, {roots, static_cast<std::size_t>(rest_coeffs), 3}, Device::CPU);
            }
            SplatData top_input(
                layout.sh_degree,
                Tensor::from_vector(means, {roots, 3}, Device::CPU),
                Tensor::from_vector(sh0, {roots, 1, 3}, Device::CPU),
                std::move(shN_tensor),
                Tensor::from_vector(scaling, {roots, 3}, Device::CPU),
                Tensor::from_vector(rotation, {roots, 4}, Device::CPU),
                Tensor::from_vector(opacity, {roots, 1}, Device::CPU),
                1.0f);

            lfs::core::BhattLodBuildOptions top_options;
            top_options.lod_base = options.lod_base;
            top_options.input_lod_opacity = true;
            top_options.leaf_input_indices = &top_leaf_bucket;
            auto top_result = lfs::core::build_bhatt_lod(top_input, top_options);
            if (!top_result) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("top-level LOD build failed: {}", top_result.error()),
                                  input_path);
            }
            top_nodes = pack_lod_output(**top_result, rest_coeffs);
            top_level_starts = relabel_level_order(top_nodes, rest_coeffs, &top_leaf_bucket);
        }

        // ------------------------------------------------------------------
        // Global assembly in BFS level order: global level L holds the top
        // tree's level-L nodes followed by every bucket's local level
        // (L - root_depth) slice. A coarse LOD cut is then a prefix of the
        // file, keeping the paged viewer's chunk working set small.
        // ------------------------------------------------------------------
        const std::size_t top_count = top_nodes.count;
        std::uint64_t total_nodes = top_count;
        if (bucket_count == 1) {
            total_nodes = summaries[0].node_count;
            summaries[0].root_depth = 0;
        } else {
            for (std::size_t b = 0; b < bucket_count; ++b) {
                total_nodes += summaries[b].node_count - 1;
            }
            const auto top_level_of = [&](const std::size_t node) {
                std::uint32_t level = 0;
                while (level + 1 < top_level_starts.size() && node >= top_level_starts[level + 1]) {
                    ++level;
                }
                return level;
            };
            for (std::size_t i = 0; i < top_count; ++i) {
                const std::uint32_t b = top_leaf_bucket[i];
                if (b != std::numeric_limits<std::uint32_t>::max()) {
                    summaries[b].root_depth = top_level_of(i);
                }
            }
        }
        LOG_INFO("ply_to_rad_lod: {} total LOD nodes ({} leaves, {} top-level)",
                 total_nodes, N, top_count);

        const auto local_levels = [&](const std::size_t b) {
            return summaries[b].level_starts.size() - 1;
        };
        const auto local_level_count = [&](const std::size_t b, const std::size_t l) -> std::size_t {
            const auto& ls = summaries[b].level_starts;
            return l + 1 < ls.size() ? ls[l + 1] - ls[l] : 0;
        };

        std::size_t global_levels = bucket_count == 1
                                        ? local_levels(0)
                                        : top_level_starts.size() - 1;
        if (bucket_count > 1) {
            for (std::size_t b = 0; b < bucket_count; ++b) {
                global_levels = std::max<std::size_t>(
                    global_levels, summaries[b].root_depth + local_levels(b));
            }
        }

        // Treelet band layout: pass 1 reads links only, ranks every node
        // within its (band, source) block, and lays blocks out band-major —
        // top tree first, then each bucket's treelets. Band 0 is the file
        // prefix, preserving the coarse-cut-as-prefix property.
        const std::size_t global_bands =
            (global_levels + kTreeletBandLevels - 1) / kTreeletBandLevels;
        const std::size_t min_local_all = bucket_count == 1 ? 0 : 1;
        TreeletRanks top_ranks;
        if (bucket_count > 1) {
            top_ranks = build_treelet_ranks(top_level_starts,
                                            top_nodes.child_count,
                                            top_nodes.child_start,
                                            0, 0, global_bands);
        }
        std::vector<TreeletRanks> bucket_ranks(bucket_count);
        for (std::size_t b = 0; b < bucket_count; ++b) {
            std::vector<std::uint16_t> link_count;
            std::vector<std::uint32_t> link_start;
            // Root-inclusive scan keeps array index == local node id (level
            // 0 is the root); min_local excludes the root from ranks when the
            // top tree carries it.
            if (!read_bucket_links(bucket_nodes_path(b), summaries[b].node_count,
                                   rest_coeffs, /*skip_root=*/false, link_count, link_start)) {
                return make_error(ErrorCode::READ_FAILURE, "Failed to scan bucket links",
                                  bucket_nodes_path(b));
            }
            bucket_ranks[b] = build_treelet_ranks(summaries[b].level_starts,
                                                  link_count, link_start,
                                                  summaries[b].root_depth,
                                                  min_local_all, global_bands);
        }
        // band_base[band][0] = top block, [band][1 + b] = bucket b's block.
        std::vector<std::vector<std::uint64_t>> band_base(
            global_bands, std::vector<std::uint64_t>(bucket_count + 1, 0));
        {
            std::uint64_t cursor = 0;
            for (std::size_t band = 0; band < global_bands; ++band) {
                band_base[band][0] = cursor;
                if (bucket_count > 1) {
                    cursor += top_ranks.band_count[band];
                }
                for (std::size_t b = 0; b < bucket_count; ++b) {
                    band_base[band][1 + b] = cursor;
                    cursor += bucket_ranks[b].band_count[band];
                }
            }
            if (cursor != total_nodes) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("treelet layout mismatch: {} != {}", cursor, total_nodes),
                                  output_path);
            }
        }
        const auto band_of_global_level = [&](const std::size_t L) {
            return L / kTreeletBandLevels;
        };
        // New global position of a top-tree node / a bucket-local node.
        const auto top_global = [&](const std::uint32_t i, const std::size_t level) {
            return band_base[band_of_global_level(level)][0] + top_ranks.rank[i];
        };
        const auto bucket_global = [&](const std::size_t b,
                                       const std::uint32_t local,
                                       const std::size_t local_level) {
            const std::size_t band =
                band_of_global_level(summaries[b].root_depth + local_level);
            return band_base[band][1 + b] + bucket_ranks[b].rank[local];
        };

        if (!report(0.78f, "Writing RAD chunks")) {
            return cancelled();
        }

        RadStreamWriter writer(output_path, total_nodes, layout.sh_degree, true,
                               options.compression_level);
        if (auto ok = writer.open(); !ok) {
            return make_error(ErrorCode::WRITE_FAILURE, ok.error(), output_path);
        }
        ChunkAccumulator accumulator(writer, rest_coeffs);

        if (bucket_count > 1) {
            // Top-tree fixups: interior children remap into the next level's
            // top block; leaves adopt their bucket root's payload and children.
            for (std::size_t i = 0; i < top_count; ++i) {
                std::uint32_t level = 0;
                while (level + 1 < top_level_starts.size() && i >= top_level_starts[level + 1]) {
                    ++level;
                }
                const std::uint32_t b = top_leaf_bucket[i];
                if (b == std::numeric_limits<std::uint32_t>::max()) {
                    if (top_nodes.child_count[i] > 0) {
                        // Sibling runs keep consecutive ranks in the child's
                        // band block, so the first child's position addresses
                        // the whole group.
                        top_nodes.child_start[i] = static_cast<std::uint32_t>(
                            top_global(top_nodes.child_start[i], level + 1));
                    }
                    continue;
                }
                const auto& summary = summaries[b];
                const float* const r = summary.root.data();
                top_nodes.means[i * 3 + 0] = r[0];
                top_nodes.means[i * 3 + 1] = r[1];
                top_nodes.means[i * 3 + 2] = r[2];
                top_nodes.rgb[i * 3 + 0] = r[3];
                top_nodes.rgb[i * 3 + 1] = r[4];
                top_nodes.rgb[i * 3 + 2] = r[5];
                top_nodes.alpha[i] = r[6];
                top_nodes.scales[i * 3 + 0] = r[7];
                top_nodes.scales[i * 3 + 1] = r[8];
                top_nodes.scales[i * 3 + 2] = r[9];
                top_nodes.rotation[i * 4 + 0] = r[10];
                top_nodes.rotation[i * 4 + 1] = r[11];
                top_nodes.rotation[i * 4 + 2] = r[12];
                top_nodes.rotation[i * 4 + 3] = r[13];
                if (rest_coeffs > 0) {
                    std::memcpy(top_nodes.shN.data() + i * static_cast<std::size_t>(rest_coeffs) * 3,
                                summary.root_shN.data(),
                                static_cast<std::size_t>(rest_coeffs) * 3 * sizeof(float));
                }
                top_nodes.child_count[i] = summary.root_child_count;
                top_nodes.child_start[i] =
                    summary.root_child_count > 0
                        ? static_cast<std::uint32_t>(
                              bucket_global(b, summary.root_child_start, 1))
                        : 0;
            }
        }

        // Per-bucket sequential readers; slices are consumed in ascending
        // local level order, so each file is read front to back.
        std::vector<FilePtr> node_files(bucket_count);
        for (std::size_t b = 0; b < bucket_count; ++b) {
            node_files[b].reset(std::fopen(bucket_nodes_path(b).string().c_str(), "rb"));
            std::uint64_t n = 0;
            if (!node_files[b] || !read_exact(node_files[b].get(), &n, sizeof(n)) ||
                n != summaries[b].node_count) {
                return make_error(ErrorCode::READ_FAILURE, "Failed to open bucket nodes",
                                  bucket_nodes_path(b));
            }
            if (bucket_count > 1) {
                // Skip the root record: the top tree carries it.
                if (std::fseek(node_files[b].get(),
                               static_cast<long>(bucket_node_record_bytes(rest_coeffs)),
                               SEEK_CUR) != 0) {
                    return make_error(ErrorCode::READ_FAILURE, "Failed to seek bucket nodes",
                                      bucket_nodes_path(b));
                }
            }
        }

        // Banded emission: buffer a source's whole band, remap child_start
        // through the treelet mapping, then emit rows in rank order. Bucket
        // files are still consumed strictly sequentially.
        const auto gather_rows = [&](const BucketNodes& src, const std::uint32_t src_base,
                                     const std::vector<std::uint32_t>& order, BucketNodes& dst) {
            const std::size_t n = order.size();
            const std::size_t sh_floats = static_cast<std::size_t>(rest_coeffs) * 3;
            dst.count = n;
            dst.means.resize(n * 3);
            dst.rgb.resize(n * 3);
            dst.alpha.resize(n);
            dst.scales.resize(n * 3);
            dst.rotation.resize(n * 4);
            dst.shN.resize(n * sh_floats);
            dst.child_count.resize(n);
            dst.child_start.resize(n);
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, n),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t k = range.begin(); k != range.end(); ++k) {
                        const std::size_t j = order[k] - src_base;
                        std::memcpy(dst.means.data() + k * 3, src.means.data() + j * 3, 12);
                        std::memcpy(dst.rgb.data() + k * 3, src.rgb.data() + j * 3, 12);
                        dst.alpha[k] = src.alpha[j];
                        std::memcpy(dst.scales.data() + k * 3, src.scales.data() + j * 3, 12);
                        std::memcpy(dst.rotation.data() + k * 4, src.rotation.data() + j * 4, 16);
                        if (sh_floats > 0) {
                            std::memcpy(dst.shN.data() + k * sh_floats,
                                        src.shN.data() + j * sh_floats, sh_floats * 4);
                        }
                        dst.child_count[k] = src.child_count[j];
                        dst.child_start[k] = src.child_start[j];
                    }
                });
        };
        // Per-band emit order (old indices sorted by rank), rebuilt per block.
        const auto block_order = [&](const TreeletRanks& ranks, const std::uint32_t lo,
                                     const std::uint32_t hi, const std::size_t count) {
            std::vector<std::uint32_t> order(count);
            for (std::uint32_t i = lo; i < hi; ++i) {
                order[ranks.rank[i]] = i;
            }
            return order;
        };

        BucketNodes band_nodes;
        BucketNodes emit_nodes;
        for (std::size_t band = 0; band < global_bands; ++band) {
            const std::size_t g_lo = band * kTreeletBandLevels;
            const std::size_t g_hi = g_lo + kTreeletBandLevels;
            if (bucket_count > 1 && top_ranks.band_count[band] > 0) {
                const std::size_t lvl_lo = g_lo;
                const std::size_t lvl_hi =
                    std::min<std::size_t>(top_level_starts.size() - 1, g_hi);
                const std::uint32_t lo = top_level_starts[lvl_lo];
                const std::uint32_t hi = top_level_starts[lvl_hi];
                const auto order = block_order(top_ranks, lo, hi,
                                               static_cast<std::size_t>(top_ranks.band_count[band]));
                gather_rows(top_nodes, 0, order, emit_nodes);
                if (auto ok = accumulator.push_span(emit_nodes, 0, emit_nodes.count); !ok) {
                    return make_error(ErrorCode::WRITE_FAILURE, ok.error(), output_path);
                }
            }
            for (std::size_t b = 0; b < bucket_count; ++b) {
                const auto& summary = summaries[b];
                if (bucket_ranks[b].band_count[band] == 0) {
                    continue;
                }
                const std::size_t depth = summary.root_depth;
                const std::size_t levels = summary.level_starts.size() - 1;
                const std::size_t l_lo =
                    std::max<std::size_t>(g_lo > depth ? g_lo - depth : 0, min_local_all);
                const std::size_t l_hi = std::min<std::size_t>(levels, g_hi - depth);
                const std::uint32_t lo = summary.level_starts[l_lo];
                const std::uint32_t hi = summary.level_starts[l_hi];
                if (!read_node_slice(node_files[b].get(), hi - lo, rest_coeffs, band_nodes)) {
                    return make_error(ErrorCode::READ_FAILURE, "Failed to read bucket node band",
                                      bucket_nodes_path(b));
                }
                // Remap child pointers: a child group lives either deeper in
                // this band or roots a next-band treelet; bucket_global covers
                // both. Child level = node level + 1, derived per row.
                tbb::parallel_for(
                    tbb::blocked_range<std::size_t>(0, band_nodes.count),
                    [&](const tbb::blocked_range<std::size_t>& range) {
                        for (std::size_t k = range.begin(); k != range.end(); ++k) {
                            if (band_nodes.child_count[k] == 0) {
                                continue;
                            }
                            const std::uint32_t local = lo + static_cast<std::uint32_t>(k);
                            const std::size_t node_level =
                                std::upper_bound(summary.level_starts.begin(),
                                                 summary.level_starts.end(), local) -
                                summary.level_starts.begin() - 1;
                            band_nodes.child_start[k] = static_cast<std::uint32_t>(
                                bucket_global(b, band_nodes.child_start[k], node_level + 1));
                        }
                    });
                const auto order = block_order(bucket_ranks[b], lo, hi,
                                               static_cast<std::size_t>(bucket_ranks[b].band_count[band]));
                gather_rows(band_nodes, lo, order, emit_nodes);
                if (auto ok = accumulator.push_span(emit_nodes, 0, emit_nodes.count); !ok) {
                    return make_error(ErrorCode::WRITE_FAILURE, ok.error(), output_path);
                }
            }
            const float p = 0.78f + 0.2f * static_cast<float>(band + 1) / static_cast<float>(global_bands);
            if (!report(p, std::format("Writing RAD chunks (band {}/{})", band + 1, global_bands))) {
                return cancelled();
            }
        }
        node_files.clear();
        for (std::size_t b = 0; b < bucket_count; ++b) {
            std::error_code ec;
            std::filesystem::remove(bucket_nodes_path(b), ec);
        }

        if (auto ok = accumulator.finish(); !ok) {
            return make_error(ErrorCode::WRITE_FAILURE, ok.error(), output_path);
        }
        if (auto ok = writer.finish(); !ok) {
            return make_error(ErrorCode::WRITE_FAILURE, ok.error(), output_path);
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - t_start);
        LOG_INFO("ply_to_rad_lod: wrote {} ({} nodes from {} splats) in {}s",
                 lfs::core::path_to_utf8(output_path), total_nodes, N, elapsed.count());

        if (!report(1.0f, "Conversion complete")) {
            return cancelled();
        }
        return {};
    }

} // namespace lfs::io
