/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU/GPU parity for the packed page-dequant path: every chunk of a
// converter-produced RAD file must decode to the same pool contents through
// decode_rad_chunk_packed + the CUDA kernel as through the CPU reference
// decode_rad_chunk_into (+ swizzle + expand_rad_meta_page). Pure-arithmetic
// encodings must match bit-exactly; libm paths (exp/log/trig) within ULPs.

#include "core/cuda/sh_layout.cuh"
#include "io/formats/rad.hpp"
#include "io/ply_to_rad_lod.hpp"
#include "rendering/lod_page_dequant_cuda.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <vector>

namespace {

    using lfs::core::SplatLodTree;

    constexpr std::size_t kPage = SplatLodTree::kChunkSplats;

    void writeSyntheticShPly(const std::filesystem::path& path, const std::size_t count) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> pos(-40.0f, 40.0f);
        std::uniform_real_distribution<float> sh(-0.4f, 0.4f);
        std::uniform_real_distribution<float> log_scale(-6.0f, -2.0f);
        std::uniform_real_distribution<float> quat(-1.0f, 1.0f);

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good());
        out << "ply\nformat binary_little_endian 1.0\n"
            << "element vertex " << count << "\n";
        for (const char* name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"}) {
            out << "property float " << name << "\n";
        }
        for (int i = 0; i < 45; ++i) {
            out << "property float f_rest_" << i << "\n";
        }
        for (const char* name : {"opacity", "scale_0", "scale_1", "scale_2",
                                 "rot_0", "rot_1", "rot_2", "rot_3"}) {
            out << "property float " << name << "\n";
        }
        out << "end_header\n";

        std::vector<float> row(62);
        for (std::size_t i = 0; i < count; ++i) {
            row[0] = pos(rng);
            row[1] = pos(rng);
            row[2] = pos(rng);
            row[3] = row[4] = row[5] = 0.0f;
            row[6] = 0.1f + sh(rng);
            row[7] = 0.2f + sh(rng);
            row[8] = 0.3f + sh(rng);
            for (int c = 0; c < 45; ++c) {
                row[9 + static_cast<std::size_t>(c)] = sh(rng) * 0.25f;
            }
            row[54] = 2.0f;
            row[55] = row[56] = row[57] = log_scale(rng);
            row[58] = 1.0f + quat(rng);
            row[59] = quat(rng) * 0.3f;
            row[60] = quat(rng) * 0.3f;
            row[61] = quat(rng) * 0.3f;
            out.write(reinterpret_cast<const char*>(row.data()),
                      static_cast<std::streamsize>(row.size() * sizeof(float)));
        }
        ASSERT_TRUE(out.good());
    }

    // Reference of the retired CPU writeSwizzledShPage: canonical rows packed
    // into float4 slots, lane-swizzled per kShReorderSize block.
    void referenceSwizzle(const std::vector<float>& canonical,
                          const std::size_t count,
                          const std::uint32_t src_rest,
                          const std::uint32_t dst_slots,
                          std::vector<float>& dst) {
        std::fill(dst.begin(), dst.end(), 0.0f);
        if (count == 0 || dst_slots == 0u || src_rest == 0u) {
            return;
        }
        const std::size_t stride = static_cast<std::size_t>(src_rest) * lfs::core::kShChannels;
        for (std::size_t i = 0; i < count; ++i) {
            const float* const row = canonical.data() + i * stride;
            const std::size_t block = i / lfs::core::kShReorderSize;
            const std::size_t lane = i % lfs::core::kShReorderSize;
            for (std::uint32_t slot = 0; slot < dst_slots; ++slot) {
                const std::size_t float4_index =
                    block * dst_slots * lfs::core::kShReorderSize +
                    static_cast<std::size_t>(slot) * lfs::core::kShReorderSize + lane;
                for (std::size_t component = 0; component < 4u; ++component) {
                    const std::size_t source_index = static_cast<std::size_t>(slot) * 4u + component;
                    dst[float4_index * 4u + component] =
                        source_index < stride ? row[source_index] : 0.0f;
                }
            }
        }
    }

    void expectClose(const std::span<const float> gpu,
                     const std::span<const float> cpu,
                     const char* const label,
                     const double rel = 1e-5,
                     const double abs = 1e-6) {
        ASSERT_EQ(gpu.size(), cpu.size()) << label;
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < gpu.size() && mismatches < 8; ++i) {
            const double tolerance = abs + rel * std::abs(static_cast<double>(cpu[i]));
            if (std::abs(static_cast<double>(gpu[i]) - static_cast<double>(cpu[i])) > tolerance) {
                ADD_FAILURE() << label << " mismatch at " << i << ": gpu=" << gpu[i]
                              << " cpu=" << cpu[i];
                ++mismatches;
            }
        }
    }

    template <typename T>
    std::vector<T> readDevice(const void* const ptr, const std::size_t count) {
        std::vector<T> host(count);
        EXPECT_EQ(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return host;
    }

    TEST(LodPageDequant, KernelMatchesCpuDecodeOnConvertedFile) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "lod_page_dequant";
        std::filesystem::remove_all(temp_dir);
        std::filesystem::create_directories(temp_dir);
        const auto ply_path = temp_dir / "dequant.ply";
        const auto rad_path = temp_dir / "dequant.rad";
        writeSyntheticShPly(ply_path, 150'000);
        lfs::io::PlyToRadLodOptions options;
        options.target_bucket_splats = 65'536;
        options.temp_dir = temp_dir / "scratch";
        ASSERT_TRUE(lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options).has_value());
        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());

        auto loaded = lfs::io::load_rad(rad_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->rad_source.valid());
        const auto& source = loaded->lod_tree->rad_source;
        const bool lod_opacity = loaded->lod_tree->lod_opacity_encoded;
        const int max_sh = loaded->get_max_sh_degree();
        ASSERT_EQ(max_sh, 3);

        auto view = lfs::io::open_rad_meta_sidecar(rad_path);
        ASSERT_TRUE(view.has_value()) << view.error();

        const std::uint32_t dst_rest = 15;
        const std::uint32_t dst_slots = lfs::core::sh_float4_slots_for_rest(dst_rest);
        const std::size_t sh_floats = lfs::core::sh_swizzled_byte_count(kPage, dst_rest) / sizeof(float);

        struct DeviceBuffer {
            void* ptr = nullptr;
            explicit DeviceBuffer(const std::size_t bytes) {
                EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
                EXPECT_EQ(cudaMemset(ptr, 0xCD, bytes), cudaSuccess);
            }
            ~DeviceBuffer() { (void)cudaFree(ptr); }
        };
        DeviceBuffer d_means(kPage * 3 * sizeof(float));
        DeviceBuffer d_sh0(kPage * 3 * sizeof(float));
        DeviceBuffer d_shN(sh_floats * sizeof(float));
        DeviceBuffer d_rot(kPage * 4 * sizeof(float));
        DeviceBuffer d_scale(kPage * 3 * sizeof(float));
        DeviceBuffer d_opacity(kPage * sizeof(float));
        DeviceBuffer d_bounds(kPage * sizeof(float4));
        DeviceBuffer d_links(kPage * sizeof(uint4));
        DeviceBuffer d_slot(64u << 20);

        lfs::vis::LodPoolDeviceView pool{};
        pool.means = static_cast<float*>(d_means.ptr);
        pool.sh0 = static_cast<float*>(d_sh0.ptr);
        pool.shN = static_cast<float*>(d_shN.ptr);
        pool.rotation = static_cast<float*>(d_rot.ptr);
        pool.scaling = static_cast<float*>(d_scale.ptr);
        pool.opacity = static_cast<float*>(d_opacity.ptr);
        pool.meta_bounds = static_cast<float4*>(d_bounds.ptr);
        pool.meta_links = static_cast<uint4*>(d_links.ptr);
        pool.dst_rest = dst_rest;
        pool.dst_slots = dst_slots;

        std::ifstream in(rad_path, std::ios::binary);
        ASSERT_TRUE(in.good());

        std::vector<std::uint8_t> chunk_bytes;
        std::vector<std::uint8_t> slot(64u << 20);
        std::vector<float> ref_means(kPage * 3), ref_sh0(kPage * 3), ref_scale(kPage * 3);
        std::vector<float> ref_rot(kPage * 4), ref_opacity(kPage);
        std::vector<float> shN_canonical;
        std::vector<float> ref_shN(sh_floats);

        // First, a middle, and the (slack-bearing) last chunk.
        std::vector<std::size_t> chunks{0};
        if (source.chunks.size() > 2) {
            chunks.push_back(source.chunks.size() / 2);
        }
        if (source.chunks.size() > 1) {
            chunks.push_back(source.chunks.size() - 1);
        }

        for (const std::size_t c : chunks) {
            const auto& range = source.chunks[c];
            chunk_bytes.resize(range.file_bytes);
            in.seekg(static_cast<std::streamoff>(range.file_offset), std::ios::beg);
            in.read(reinterpret_cast<char*>(chunk_bytes.data()),
                    static_cast<std::streamsize>(chunk_bytes.size()));
            ASSERT_TRUE(in.good());

            const lfs::io::RadChunkDsts dsts{
                .means = ref_means.data(),
                .opacity_raw = ref_opacity.data(),
                .sh0_raw = ref_sh0.data(),
                .scaling_raw = ref_scale.data(),
                .rotation_raw = ref_rot.data(),
                .shN_canonical = &shN_canonical,
            };
            auto info = lfs::io::decode_rad_chunk_into(
                std::span<const std::uint8_t>(chunk_bytes), max_sh, lod_opacity, kPage, dsts);
            ASSERT_TRUE(info.has_value()) << info.error();
            const std::size_t count = static_cast<std::size_t>(info->count);
            referenceSwizzle(shN_canonical, count, info->sh_coeffs_rest, dst_slots, ref_shN);

            auto desc = lfs::io::decode_rad_chunk_packed(
                std::span<const std::uint8_t>(chunk_bytes), max_sh, lod_opacity, kPage,
                *view, static_cast<std::uint32_t>(c), std::span<std::uint8_t>(slot));
            ASSERT_TRUE(desc.has_value()) << desc.error();
            ASSERT_EQ(desc->count, info->count);

            ASSERT_EQ(cudaMemcpy(d_slot.ptr, slot.data(), desc->used_bytes,
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            ASSERT_EQ(lfs::vis::launchLodPageDequant(
                          static_cast<const std::uint8_t*>(d_slot.ptr), *desc, pool,
                          /*page=*/0, static_cast<std::uint32_t>(kPage), nullptr),
                      cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            const auto gpu_means = readDevice<float>(d_means.ptr, count * 3);
            for (std::size_t i = 0; i < count * 3; ++i) {
                ASSERT_EQ(gpu_means[i], ref_means[i]) << "means bit-parity at " << i;
            }
            expectClose(readDevice<float>(d_sh0.ptr, count * 3),
                        std::span<const float>(ref_sh0.data(), count * 3), "sh0");
            expectClose(readDevice<float>(d_scale.ptr, count * 3),
                        std::span<const float>(ref_scale.data(), count * 3), "scaling");
            expectClose(readDevice<float>(d_opacity.ptr, count),
                        std::span<const float>(ref_opacity.data(), count), "opacity");
            // Reference rotation layout is (w,x,y,z) in pool order already.
            expectClose(readDevice<float>(d_rot.ptr, count * 4),
                        std::span<const float>(ref_rot.data(), count * 4), "rotation",
                        1e-4, 1e-5);
            expectClose(readDevice<float>(d_shN.ptr, sh_floats),
                        std::span<const float>(ref_shN.data(), sh_floats), "shN");

            const std::size_t logical_start = c * kPage;
            const std::size_t run = std::min(kPage, view->node_count - logical_start);
            std::vector<lfs::core::NodeBoundsRecord> ref_bounds(run);
            std::vector<lfs::core::NodeLinksRecord> ref_links(run);
            lfs::io::expand_rad_meta_page(*view, static_cast<std::uint32_t>(c), run,
                                          ref_bounds.data(), ref_links.data());
            const auto gpu_bounds = readDevice<float4>(d_bounds.ptr, kPage);
            const auto gpu_links = readDevice<uint4>(d_links.ptr, kPage);
            // Host code contracts bbox_min + q*inv*extent into FMA, the
            // kernel does not (--fmad=false); with the cancellation in the
            // sum that costs up to ~1 ULP at the extent's magnitude.
            const auto& frame = view->chunks[c];
            const float center_tol[3] = {
                std::abs(frame.bbox_extent[0]) * 1e-6f + 1e-7f,
                std::abs(frame.bbox_extent[1]) * 1e-6f + 1e-7f,
                std::abs(frame.bbox_extent[2]) * 1e-6f + 1e-7f,
            };
            for (std::size_t i = 0; i < run; ++i) {
                EXPECT_EQ(gpu_links[i].x, ref_links[i].child_start);
                EXPECT_EQ(gpu_links[i].y, ref_links[i].packed);
                EXPECT_EQ(gpu_links[i].z, ref_links[i].parent);
                EXPECT_EQ(gpu_links[i].w, ref_links[i].logical);
                EXPECT_NEAR(gpu_bounds[i].x, ref_bounds[i].x, center_tol[0]);
                EXPECT_NEAR(gpu_bounds[i].y, ref_bounds[i].y, center_tol[1]);
                EXPECT_NEAR(gpu_bounds[i].z, ref_bounds[i].z, center_tol[2]);
                EXPECT_NEAR(gpu_bounds[i].w, ref_bounds[i].size,
                            std::abs(ref_bounds[i].size) * 1e-5f);
            }
            for (std::size_t i = run; i < kPage; ++i) {
                EXPECT_EQ(gpu_links[i].w, 0xFFFFFFFFu) << "slack link sentinel at " << i;
                EXPECT_EQ(gpu_bounds[i].w, 0.0f) << "slack bounds at " << i;
            }
        }
    }

    // CPU sink-cost comparison on a real converted file: full decode (old
    // sink) vs inflate-only packed decode (new sink). Informational; run with
    // LFS_RAD_BENCH_FILE=<file.rad> pointing at a RAD with a .rad.meta.
    TEST(LodPageDequant, SinkCpuCostBenchmark) {
        const char* const bench = std::getenv("LFS_RAD_BENCH_FILE");
        if (bench == nullptr) {
            GTEST_SKIP() << "set LFS_RAD_BENCH_FILE to run";
        }
        const std::filesystem::path rad_path(bench);
        auto loaded = lfs::io::load_rad(rad_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->rad_source.valid());
        const auto& source = loaded->lod_tree->rad_source;
        auto view = lfs::io::open_rad_meta_sidecar(rad_path);
        ASSERT_TRUE(view.has_value()) << view.error();
        const bool lod_opacity = loaded->lod_tree->lod_opacity_encoded;
        const int max_sh = loaded->get_max_sh_degree();

        const std::size_t n = std::min<std::size_t>(source.chunks.size(), 256);
        std::ifstream in(rad_path, std::ios::binary);
        std::vector<std::vector<std::uint8_t>> raw(n);
        for (std::size_t c = 0; c < n; ++c) {
            raw[c].resize(source.chunks[c].file_bytes);
            in.seekg(static_cast<std::streamoff>(source.chunks[c].file_offset), std::ios::beg);
            in.read(reinterpret_cast<char*>(raw[c].data()),
                    static_cast<std::streamsize>(raw[c].size()));
            ASSERT_TRUE(in.good());
        }

        // Touch the sidecar planes once so neither loop measures cold mmap
        // page faults; both the old sink (expand_rad_meta_page) and the new
        // one (plane memcpy) read the same bytes in production.
        std::uint64_t warm = 0;
        for (std::size_t c = 0; c < n; ++c) {
            const std::size_t start = c * kPage;
            const std::size_t run = std::min(kPage, view->node_count - start);
            const auto* const b = reinterpret_cast<const std::uint8_t*>(view->bounds + start);
            const auto* const l = reinterpret_cast<const std::uint8_t*>(view->links + start);
            for (std::size_t i = 0; i < run * sizeof(lfs::core::RadMetaBoundsQ); i += 4096) {
                warm += b[i];
            }
            for (std::size_t i = 0; i < run * sizeof(lfs::core::RadMetaLinksQ); i += 4096) {
                warm += l[i];
            }
        }
        ASSERT_GE(warm, 0u);

        std::vector<float> means(kPage * 3), sh0(kPage * 3), scale(kPage * 3);
        std::vector<float> rot(kPage * 4), opacity(kPage);
        std::vector<float> shN_canonical;
        std::vector<float> shN_swizzled(
            lfs::core::sh_swizzled_byte_count(kPage, 15) / sizeof(float));
        std::vector<lfs::core::NodeBoundsRecord> bounds(kPage);
        std::vector<lfs::core::NodeLinksRecord> links(kPage);
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t c = 0; c < n; ++c) {
            const lfs::io::RadChunkDsts dsts{
                .means = means.data(),
                .opacity_raw = opacity.data(),
                .sh0_raw = sh0.data(),
                .scaling_raw = scale.data(),
                .rotation_raw = rot.data(),
                .shN_canonical = &shN_canonical,
            };
            auto info = lfs::io::decode_rad_chunk_into(
                std::span<const std::uint8_t>(raw[c]), max_sh, lod_opacity, kPage, dsts);
            ASSERT_TRUE(info.has_value()) << info.error();
            // The old sink's remaining CPU work: SH swizzle + meta expansion.
            referenceSwizzle(shN_canonical, static_cast<std::size_t>(info->count),
                             info->sh_coeffs_rest,
                             lfs::core::sh_float4_slots_for_rest(15), shN_swizzled);
            std::memset(bounds.data(), 0, bounds.size() * sizeof(bounds[0]));
            std::memset(links.data(), 0xFF, links.size() * sizeof(links[0]));
            const std::size_t start = c * kPage;
            const std::size_t run = std::min(kPage, view->node_count - start);
            lfs::io::expand_rad_meta_page(*view, static_cast<std::uint32_t>(c), run,
                                          bounds.data(), links.data());
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> slot(64u << 20);
        for (std::size_t c = 0; c < n; ++c) {
            auto desc = lfs::io::decode_rad_chunk_packed(
                std::span<const std::uint8_t>(raw[c]), max_sh, lod_opacity, kPage,
                *view, static_cast<std::uint32_t>(c), std::span<std::uint8_t>(slot));
            ASSERT_TRUE(desc.has_value()) << desc.error();
        }
        const auto t2 = std::chrono::steady_clock::now();

        const auto ms = [](const auto a, const auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double full_ms = ms(t0, t1) / static_cast<double>(n);
        const double packed_ms = ms(t1, t2) / static_cast<double>(n);
        std::printf("sink CPU cost over %zu chunks: full decode %.3f ms/chunk "
                    "(%.0f chunks/s/core), packed %.3f ms/chunk (%.0f chunks/s/core)\n",
                    n, full_ms, 1000.0 / full_ms, packed_ms, 1000.0 / packed_ms);
    }

} // namespace
