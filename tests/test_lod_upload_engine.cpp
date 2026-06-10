/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/lod_upload_engine.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <thread>
#include <vector>

namespace {

    using lfs::vis::LodPageCache;
    using lfs::vis::LodUploadEngine;

    constexpr std::size_t kPageSplats = LodPageCache::kChunkSplats;

    struct DevicePool {
        void* base = nullptr;
        void* meta_base = nullptr;
        std::array<std::size_t, 6> region_offset{};
        std::size_t splat_capacity = 0;
        std::size_t meta_links_offset = 0;

        explicit DevicePool(const std::size_t pages, const std::uint32_t dst_rest) {
            splat_capacity = pages * kPageSplats;
            std::array<std::size_t, 6> region_bytes{};
            region_bytes[0] = splat_capacity * 3u * sizeof(float);
            region_bytes[1] = splat_capacity * 3u * sizeof(float);
            region_bytes[2] = dst_rest == 0u
                                  ? 4u * sizeof(float)
                                  : lfs::core::sh_swizzled_byte_count(splat_capacity, dst_rest);
            region_bytes[3] = splat_capacity * 4u * sizeof(float);
            region_bytes[4] = splat_capacity * 3u * sizeof(float);
            region_bytes[5] = splat_capacity * sizeof(float);
            std::size_t offset = 0;
            for (std::size_t i = 0; i < region_bytes.size(); ++i) {
                region_offset[i] = offset;
                offset += (region_bytes[i] + 255u) & ~std::size_t{255u};
            }
            EXPECT_EQ(cudaMalloc(&base, offset), cudaSuccess);
            EXPECT_EQ(cudaMemset(base, 0, offset), cudaSuccess);

            meta_links_offset = splat_capacity * sizeof(lfs::core::NodeBoundsRecord);
            const std::size_t meta_bytes =
                meta_links_offset + splat_capacity * sizeof(lfs::core::NodeLinksRecord);
            EXPECT_EQ(cudaMalloc(&meta_base, meta_bytes), cudaSuccess);
            EXPECT_EQ(cudaMemset(meta_base, 0, meta_bytes), cudaSuccess);
        }
        ~DevicePool() {
            if (base != nullptr) {
                (void)cudaFree(base);
            }
            if (meta_base != nullptr) {
                (void)cudaFree(meta_base);
            }
        }

        [[nodiscard]] LodUploadEngine::DeviceLayout layout(const std::uint32_t dst_rest) const {
            return {
                .device_base = base,
                .region_offset = region_offset,
                .splat_capacity = splat_capacity,
                .dst_rest = dst_rest,
                .dst_slots = lfs::core::sh_float4_slots_for_rest(dst_rest),
                .meta_base = meta_base,
                .meta_bounds_offset = 0,
                .meta_links_offset = meta_links_offset,
            };
        }
    };

    // Packs a recognizable page into an acquired slot and submits it.
    void packAndSubmit(LodUploadEngine& engine,
                       const std::uint32_t page,
                       const std::uint32_t chunk,
                       const std::size_t count) {
        auto* const slot = engine.acquireStagingSlot();
        ASSERT_NE(slot, nullptr);
        const auto staging = engine.stagingLayout();
        auto* const means = reinterpret_cast<float*>(slot->data + staging.means);
        std::iota(means, means + count * 3u, static_cast<float>(chunk) * 1000.0f);
        auto* const opacity = reinterpret_cast<float*>(slot->data + staging.opacity);
        std::iota(opacity, opacity + count, static_cast<float>(chunk));
        auto* const links =
            reinterpret_cast<lfs::core::NodeLinksRecord*>(slot->data + staging.meta_links);
        for (std::size_t i = 0; i < kPageSplats; ++i) {
            links[i] = {.child_start = chunk, .packed = 0, .parent = 0, .logical = static_cast<std::uint32_t>(i)};
        }
        engine.submitPackedPage(slot, page, chunk, 1, count);
    }

    std::vector<LodPageCache::PendingUpload> collectAll(LodUploadEngine& engine,
                                                        const std::size_t expected) {
        std::vector<LodPageCache::PendingUpload> published;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (published.size() < expected && std::chrono::steady_clock::now() < deadline) {
            auto batch = engine.collectPublished();
            for (auto& upload : batch) {
                published.push_back(std::move(upload));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return published;
    }

    TEST(LodUploadEngine, RoundTripUploadsPayloadAndMetadata) {
        DevicePool pool(4, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool.layout(0), nullptr);
        ASSERT_TRUE(engine.configured());

        constexpr std::size_t kCount = 1000;
        packAndSubmit(engine, 2, 7, kCount);
        const auto published = collectAll(engine, 1);
        ASSERT_EQ(published.size(), 1u);
        EXPECT_TRUE(published.front().error.empty()) << published.front().error;
        EXPECT_EQ(published.front().chunk, 7u);
        EXPECT_EQ(engine.lastPublishedSignalValue(), 1u);

        const std::size_t dst_start = 2u * kPageSplats;
        std::vector<float> means(kCount * 3u);
        ASSERT_EQ(cudaMemcpy(means.data(),
                             static_cast<std::uint8_t*>(pool.base) + pool.region_offset[0] +
                                 dst_start * 3u * sizeof(float),
                             means.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_FLOAT_EQ(means[0], 7000.0f);
        EXPECT_FLOAT_EQ(means.back(), 7000.0f + static_cast<float>(kCount * 3u - 1));

        std::vector<lfs::core::NodeLinksRecord> links(4);
        ASSERT_EQ(cudaMemcpy(links.data(),
                             static_cast<std::uint8_t*>(pool.meta_base) + pool.meta_links_offset +
                                 dst_start * sizeof(lfs::core::NodeLinksRecord),
                             links.size() * sizeof(lfs::core::NodeLinksRecord),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < links.size(); ++i) {
            EXPECT_EQ(links[i].child_start, 7u);
            EXPECT_EQ(links[i].logical, i);
        }
    }

    TEST(LodUploadEngine, DrainAndSyncCompletesInFlightWork) {
        DevicePool pool(8, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool.layout(0), nullptr);

        for (std::uint32_t i = 0; i < 8; ++i) {
            packAndSubmit(engine, i, 100 + i, kPageSplats);
        }
        const auto drained = engine.drainAndSync();
        EXPECT_EQ(drained.size(), 8u);
        for (const auto& upload : drained) {
            EXPECT_TRUE(upload.error.empty()) << upload.error;
        }
        EXPECT_TRUE(engine.idle());
    }

    TEST(LodUploadEngine, ReconfigureDrainsPriorLayout) {
        DevicePool pool_a(2, 0);
        DevicePool pool_b(2, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool_a.layout(0), nullptr);

        packAndSubmit(engine, 1, 5, 256);
        const auto drained = engine.configure(pool_b.layout(0), nullptr);
        EXPECT_EQ(drained.size(), 1u);
        EXPECT_TRUE(engine.idle());

        packAndSubmit(engine, 0, 6, 256);
        const auto published = collectAll(engine, 1);
        ASSERT_EQ(published.size(), 1u);
        EXPECT_TRUE(published.front().error.empty()) << published.front().error;
    }

    TEST(LodUploadEngine, MultiWorkerSignalsStayMonotone) {
        DevicePool pool(8, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool.layout(0), nullptr);

        constexpr std::size_t kWorkers = 4;
        constexpr std::size_t kPerWorker = 16;
        std::atomic<std::uint32_t> next_chunk{0};
        std::vector<std::thread> workers;
        workers.reserve(kWorkers);
        for (std::size_t w = 0; w < kWorkers; ++w) {
            workers.emplace_back([&] {
                for (std::size_t i = 0; i < kPerWorker; ++i) {
                    const std::uint32_t chunk = next_chunk.fetch_add(1);
                    packAndSubmit(engine, chunk % 8u, chunk, 1024);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        const auto drained = engine.drainAndSync();
        EXPECT_EQ(drained.size(), kWorkers * kPerWorker);
        for (const auto& upload : drained) {
            EXPECT_TRUE(upload.error.empty()) << upload.error;
        }
        EXPECT_EQ(engine.lastPublishedSignalValue(), kWorkers * kPerWorker);
    }

    TEST(LodUploadEngine, UnconfiguredAcquireReturnsNull) {
        LodUploadEngine engine;
        EXPECT_EQ(engine.acquireStagingSlot(), nullptr);
        EXPECT_TRUE(engine.idle());
        EXPECT_TRUE(engine.collectPublished().empty());
    }

} // namespace
