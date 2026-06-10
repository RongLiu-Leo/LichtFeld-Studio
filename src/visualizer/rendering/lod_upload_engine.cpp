/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lod_upload_engine.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <thread>
#include <utility>

namespace lfs::vis {
    namespace {

        constexpr std::size_t kPageSplats = LodPageCache::kChunkSplats;

        std::size_t stagingRingDepth() {
            if (const char* const env = std::getenv("LFS_LOD_STAGING_SLOTS")) {
                char* end = nullptr;
                const auto parsed = std::strtoull(env, &end, 10);
                if (end != env && parsed > 0) {
                    return static_cast<std::size_t>(parsed);
                }
            }
            const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
            return std::max<std::size_t>(8, std::clamp<std::size_t>(hw / 2, 2, 8) + 2);
        }

        LodUploadEngine::StagingLayout stagingLayoutFor(const std::uint32_t dst_rest,
                                                        const bool has_meta) {
            LodUploadEngine::StagingLayout layout{};
            const std::size_t page_sh_bytes =
                dst_rest == 0u ? 0u : lfs::core::sh_swizzled_byte_count(kPageSplats, dst_rest);
            std::size_t offset = 0;
            const auto place = [&offset](const std::size_t bytes) {
                const std::size_t at = offset;
                offset += bytes;
                return at;
            };
            layout.means = place(kPageSplats * 3u * sizeof(float));
            layout.sh0 = place(kPageSplats * 3u * sizeof(float));
            layout.shN = place(page_sh_bytes);
            layout.rotation = place(kPageSplats * 4u * sizeof(float));
            layout.scaling = place(kPageSplats * 3u * sizeof(float));
            layout.opacity = place(kPageSplats * sizeof(float));
            if (has_meta) {
                layout.meta_bounds = place(kPageSplats * sizeof(lfs::core::NodeBoundsRecord));
                layout.meta_links = place(kPageSplats * sizeof(lfs::core::NodeLinksRecord));
            }
            layout.total_bytes = offset;
            layout.page_sh_bytes = page_sh_bytes;
            return layout;
        }

        [[nodiscard]] std::string copyAsync(const void* const src,
                                            void* const dst,
                                            const std::size_t bytes,
                                            const cudaStream_t stream,
                                            const std::string_view label) {
            if (bytes == 0) {
                return {};
            }
            const cudaError_t status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess) {
                return std::format("LOD page {} H2D copy failed: {} ({})",
                                   label,
                                   cudaGetErrorName(status),
                                   cudaGetErrorString(status));
            }
            return {};
        }

    } // namespace

    LodUploadEngine::LodUploadEngine() = default;

    LodUploadEngine::~LodUploadEngine() {
        (void)drainAndSync();
        std::lock_guard lock(mutex_);
        shutdown_ = true;
        slot_cv_.notify_all();
        for (const cudaEvent_t event : event_pool_) {
            (void)cudaEventDestroy(event);
        }
        event_pool_.clear();
        releaseStagingRingLocked();
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    std::vector<LodPageCache::PendingUpload>
    LodUploadEngine::configure(const DeviceLayout& layout,
                               const lfs::rendering::CudaTimelineSemaphore* const timeline) {
        {
            std::lock_guard lock(mutex_);
            if (layout_ == layout && timeline_ == timeline) {
                return {};
            }
        }
        auto drained = drainAndSync();
        std::lock_guard lock(mutex_);
        layout_ = layout;
        timeline_ = timeline;
        if (stream_ == nullptr && layout_.valid()) {
            if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
                stream_ = nullptr;
            }
        }
        releaseStagingRingLocked();
        staging_layout_ = stagingLayoutFor(layout_.dst_rest, layout_.meta_base != nullptr);
        if (layout_.valid() && stream_ != nullptr) {
            staging_ring_.resize(stagingRingDepth());
            for (auto& slot : staging_ring_) {
                if (cudaHostAlloc(reinterpret_cast<void**>(&slot.data),
                                  staging_layout_.total_bytes,
                                  cudaHostAllocDefault) != cudaSuccess) {
                    slot.data = nullptr;
                }
                if (cudaEventCreateWithFlags(&slot.last_use, cudaEventDisableTiming) != cudaSuccess) {
                    slot.last_use = nullptr;
                }
            }
        }
        slot_cv_.notify_all();
        return drained;
    }

    bool LodUploadEngine::configured() const {
        std::lock_guard lock(mutex_);
        return layout_.valid() && stream_ != nullptr && !staging_ring_.empty();
    }

    LodUploadEngine::StagingLayout LodUploadEngine::stagingLayout() const {
        std::lock_guard lock(mutex_);
        return staging_layout_;
    }

    bool LodUploadEngine::idle() const {
        std::lock_guard lock(mutex_);
        if (!in_flight_.empty()) {
            return false;
        }
        return std::none_of(staging_ring_.begin(), staging_ring_.end(),
                            [](const StagingSlot& slot) { return slot.acquired; });
    }

    std::uint64_t LodUploadEngine::lastPublishedSignalValue() const {
        std::lock_guard lock(mutex_);
        return last_published_signal_;
    }

    LodUploadEngine::StagingSlot* LodUploadEngine::acquireStagingSlot() {
        std::unique_lock lock(mutex_);
        while (true) {
            if (shutdown_ || staging_ring_.empty()) {
                return nullptr;
            }
            StagingSlot* candidate = nullptr;
            for (std::size_t probe = 0; probe < staging_ring_.size(); ++probe) {
                StagingSlot& slot = staging_ring_[(staging_cursor_ + probe) % staging_ring_.size()];
                if (!slot.acquired && slot.data != nullptr && slot.last_use != nullptr) {
                    candidate = &slot;
                    staging_cursor_ = (staging_cursor_ + probe + 1) % staging_ring_.size();
                    break;
                }
            }
            if (candidate == nullptr) {
                slot_cv_.wait(lock);
                continue;
            }
            candidate->acquired = true;
            if (candidate->used) {
                // The slot's previous copies may still be in flight; wait off
                // the lock so other workers keep packing.
                const cudaEvent_t guard = candidate->last_use;
                lock.unlock();
                if (cudaEventSynchronize(guard) != cudaSuccess) {
                    lock.lock();
                    candidate->acquired = false;
                    slot_cv_.notify_all();
                    return nullptr;
                }
                lock.lock();
            }
            return candidate;
        }
    }

    void LodUploadEngine::releaseSlot(StagingSlot* const slot) {
        if (slot == nullptr) {
            return;
        }
        std::lock_guard lock(mutex_);
        slot->acquired = false;
        slot_cv_.notify_all();
    }

    void LodUploadEngine::submitPackedPage(StagingSlot* const slot,
                                           const std::uint32_t page,
                                           const std::uint32_t chunk,
                                           const std::uint64_t generation,
                                           const std::size_t splat_count) {
        Job job{
            .upload = {
                .page = page,
                .chunk = chunk,
                .generation = generation,
                .error = {},
            },
        };

        // One lock section covers copy-issue + timeline signal + event record
        // so timeline values stay monotone in stream order across workers.
        std::lock_guard lock(mutex_);
        const auto submit = [&]() -> std::string {
            if (!layout_.valid() || stream_ == nullptr) {
                return "LOD upload engine is not configured";
            }
            const std::size_t dst_start = static_cast<std::size_t>(page) * kPageSplats;
            const std::size_t count =
                std::min({splat_count, kPageSplats,
                          layout_.splat_capacity > dst_start ? layout_.splat_capacity - dst_start
                                                             : std::size_t{0}});
            if (count == 0) {
                return std::format("LOD upload page {} exceeds splat capacity {}",
                                   page, layout_.splat_capacity);
            }
            auto* const base = slot->data;
            auto* const device_base = static_cast<std::uint8_t*>(layout_.device_base);
            const auto region = [&](const std::size_t index) {
                return device_base + layout_.region_offset[index];
            };
            const std::size_t sh_block_offset =
                (dst_start / lfs::core::kShReorderSize) *
                static_cast<std::size_t>(layout_.dst_slots) *
                lfs::core::kShReorderSize * 4u * sizeof(float);

            std::string error =
                copyAsync(base + staging_layout_.means, region(0) + dst_start * 3u * sizeof(float),
                          count * 3u * sizeof(float), stream_, "means");
            if (error.empty()) {
                error = copyAsync(base + staging_layout_.sh0, region(1) + dst_start * 3u * sizeof(float),
                                  count * 3u * sizeof(float), stream_, "sh0");
            }
            if (error.empty() && staging_layout_.page_sh_bytes > 0) {
                error = copyAsync(base + staging_layout_.shN, region(2) + sh_block_offset,
                                  staging_layout_.page_sh_bytes, stream_, "shN");
            }
            if (error.empty()) {
                error = copyAsync(base + staging_layout_.rotation, region(3) + dst_start * 4u * sizeof(float),
                                  count * 4u * sizeof(float), stream_, "rotation");
            }
            if (error.empty()) {
                error = copyAsync(base + staging_layout_.scaling, region(4) + dst_start * 3u * sizeof(float),
                                  count * 3u * sizeof(float), stream_, "scaling");
            }
            if (error.empty()) {
                error = copyAsync(base + staging_layout_.opacity, region(5) + dst_start * sizeof(float),
                                  count * sizeof(float), stream_, "opacity");
            }
            if (error.empty() && layout_.meta_base != nullptr) {
                auto* const meta_base = static_cast<std::uint8_t*>(layout_.meta_base);
                error = copyAsync(base + staging_layout_.meta_bounds,
                                  meta_base + layout_.meta_bounds_offset +
                                      dst_start * sizeof(lfs::core::NodeBoundsRecord),
                                  kPageSplats * sizeof(lfs::core::NodeBoundsRecord),
                                  stream_, "node bounds");
                if (error.empty()) {
                    error = copyAsync(base + staging_layout_.meta_links,
                                      meta_base + layout_.meta_links_offset +
                                          dst_start * sizeof(lfs::core::NodeLinksRecord),
                                      kPageSplats * sizeof(lfs::core::NodeLinksRecord),
                                      stream_, "node links");
                }
            }
            if (!error.empty()) {
                return error;
            }

            const cudaEvent_t event = acquireEventLocked();
            if (event == nullptr) {
                return "LOD upload failed to create a CUDA event";
            }
            const std::uint64_t signal_value = ++signal_counter_;
            if (timeline_ != nullptr && timeline_->valid()) {
                (void)timeline_->cudaSignal(signal_value, stream_);
            }
            if (const cudaError_t status = cudaEventRecord(event, stream_); status != cudaSuccess) {
                event_pool_.push_back(event);
                return std::format("LOD upload event record failed: {} ({})",
                                   cudaGetErrorName(status),
                                   cudaGetErrorString(status));
            }
            (void)cudaEventRecord(slot->last_use, stream_);
            slot->used = true;
            job.event = event;
            job.signal_value = signal_value;
            return {};
        };

        job.upload.error = submit();
        slot->acquired = false;
        slot_cv_.notify_all();
        in_flight_.push_back(std::move(job));
    }

    cudaEvent_t LodUploadEngine::acquireEventLocked() {
        if (!event_pool_.empty()) {
            const cudaEvent_t event = event_pool_.back();
            event_pool_.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            return nullptr;
        }
        return event;
    }

    std::vector<LodPageCache::PendingUpload>
    LodUploadEngine::takeCompletedLocked(const bool wait_for_all) {
        std::vector<LodPageCache::PendingUpload> published;
        while (!in_flight_.empty()) {
            Job& job = in_flight_.front();
            if (job.event != nullptr) {
                if (!wait_for_all) {
                    const cudaError_t status = cudaEventQuery(job.event);
                    if (status == cudaErrorNotReady) {
                        break;
                    }
                    if (status != cudaSuccess) {
                        job.upload.error = std::format("LOD upload event query failed: {} ({})",
                                                       cudaGetErrorName(status),
                                                       cudaGetErrorString(status));
                    }
                }
                event_pool_.push_back(job.event);
                job.event = nullptr;
                last_published_signal_ = std::max(last_published_signal_, job.signal_value);
            }
            published.push_back(std::move(job.upload));
            in_flight_.pop_front();
        }
        return published;
    }

    std::vector<LodPageCache::PendingUpload> LodUploadEngine::collectPublished() {
        std::lock_guard lock(mutex_);
        return takeCompletedLocked(false);
    }

    std::vector<LodPageCache::PendingUpload> LodUploadEngine::drainAndSync() {
        {
            std::unique_lock lock(mutex_);
            slot_cv_.wait(lock, [this] {
                return std::none_of(staging_ring_.begin(), staging_ring_.end(),
                                    [](const StagingSlot& slot) { return slot.acquired; });
            });
        }
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
        }
        std::lock_guard lock(mutex_);
        return takeCompletedLocked(true);
    }

    void LodUploadEngine::releaseStagingRingLocked() {
        for (auto& slot : staging_ring_) {
            if (slot.data != nullptr) {
                (void)cudaFreeHost(slot.data);
            }
            if (slot.last_use != nullptr) {
                (void)cudaEventDestroy(slot.last_use);
            }
        }
        staging_ring_.clear();
        staging_cursor_ = 0;
    }

} // namespace lfs::vis
