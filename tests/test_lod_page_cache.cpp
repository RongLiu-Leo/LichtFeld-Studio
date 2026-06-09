/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/lod_page_cache.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

    using lfs::vis::LodPageCache;

    constexpr std::uint32_t kInvalid = LodPageCache::kInvalidPage;

    std::vector<LodPageCache::PendingUpload> drainAndComplete(LodPageCache& cache) {
        auto uploads = cache.drainPendingUploads();
        cache.completeUploads(uploads);
        return uploads;
    }

    TEST(LodPageCache, ConfigureBoundedPinsRootAndStaysPartial) {
        LodPageCache cache;
        cache.configure(100, 10, 1);

        const auto& snapshot = cache.snapshot();
        EXPECT_EQ(snapshot.logical_chunks, 100u);
        EXPECT_EQ(snapshot.physical_pages, 10u);
        EXPECT_TRUE(cache.configured());
        EXPECT_FALSE(cache.fullyResident());
        EXPECT_EQ(snapshot.chunk_to_page[0], 0u);
        EXPECT_EQ(snapshot.page_to_chunk[0], 0u);
        for (std::uint32_t chunk = 1; chunk < 100; ++chunk) {
            EXPECT_EQ(snapshot.chunk_to_page[chunk], kInvalid);
        }
    }

    TEST(LodPageCache, ConfigureFullCapacityIsFullyResident) {
        LodPageCache cache;
        cache.configure(16, 0, 1);

        EXPECT_EQ(cache.snapshot().physical_pages, 16u);
        EXPECT_TRUE(cache.fullyResident());
        for (std::uint32_t chunk = 0; chunk < 16; ++chunk) {
            EXPECT_EQ(cache.snapshot().chunk_to_page[chunk], chunk);
        }
    }

    TEST(LodPageCache, SubmitTraversalPriorityRespectsRequestBudget) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        (void)drainAndComplete(cache); // root bootstrap upload

        std::vector<std::uint32_t> wanted(40);
        std::iota(wanted.begin(), wanted.end(), 10u);
        cache.submitTraversalPriority(wanted);

        // Budget for 10 pages: clamp(10/4, min(8,10), min(64,10)) = 8.
        const auto uploads = cache.drainPendingUploads();
        EXPECT_EQ(uploads.size(), 8u);
        for (std::size_t i = 0; i < uploads.size(); ++i) {
            EXPECT_EQ(uploads[i].chunk, wanted[i]) << "caller priority order must be preserved";
        }
    }

    TEST(LodPageCache, CompleteUploadsPublishesResidency) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        (void)drainAndComplete(cache);

        const std::uint64_t generation_before = cache.snapshot().generation;
        const std::vector<std::uint32_t> wanted{42, 43};
        cache.submitTraversalPriority(wanted);
        EXPECT_TRUE(cache.hasOutstandingWork());
        (void)drainAndComplete(cache);

        EXPECT_FALSE(cache.hasOutstandingWork());
        EXPECT_NE(cache.snapshot().chunk_to_page[42], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[43], kInvalid);
        EXPECT_GT(cache.snapshot().generation, generation_before);
    }

    TEST(LodPageCache, ProtectedChunksSurviveEvictionPressure) {
        LodPageCache cache;
        cache.configure(100, 4, 1);
        (void)drainAndComplete(cache);

        std::vector<std::uint32_t> first{10, 11, 12};
        cache.submitTraversalPriority(first);
        (void)drainAndComplete(cache);
        ASSERT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        ASSERT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
        ASSERT_NE(cache.snapshot().chunk_to_page[12], kInvalid);

        // The pool is now full (root + 3). Requesting new chunks while
        // protecting 10 and 11 may only evict chunk 12.
        std::vector<std::uint32_t> next{20};
        std::vector<std::uint32_t> protected_chunks{10, 11};
        cache.submitTraversalPriority(next, protected_chunks);
        (void)drainAndComplete(cache);

        EXPECT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
        EXPECT_EQ(cache.snapshot().chunk_to_page[12], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[20], kInvalid);
    }

    TEST(LodPageCache, PinnedRootIsNeverEvicted) {
        LodPageCache cache;
        cache.configure(100, 2, 1);
        (void)drainAndComplete(cache);

        for (std::uint32_t chunk = 10; chunk < 30; ++chunk) {
            std::vector<std::uint32_t> wanted{chunk};
            cache.submitTraversalPriority(wanted);
            (void)drainAndComplete(cache);
        }

        EXPECT_EQ(cache.snapshot().chunk_to_page[0], 0u) << "pinned root must stay resident";
        EXPECT_NE(cache.snapshot().chunk_to_page[29], kInvalid);
    }

    TEST(LodPageCache, LruEvictsLeastRecentlyTouchedChunk) {
        LodPageCache cache;
        cache.configure(100, 3, 1);
        (void)drainAndComplete(cache);

        std::vector<std::uint32_t> first{10, 11};
        cache.submitTraversalPriority(first);
        (void)drainAndComplete(cache);

        // Refresh chunk 10's LRU clock, then force one eviction.
        std::vector<std::uint32_t> refresh{10, 20};
        cache.submitTraversalPriority(refresh);
        (void)drainAndComplete(cache);

        EXPECT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        EXPECT_EQ(cache.snapshot().chunk_to_page[11], kInvalid) << "stale chunk must be evicted first";
        EXPECT_NE(cache.snapshot().chunk_to_page[20], kInvalid);
    }

} // namespace
