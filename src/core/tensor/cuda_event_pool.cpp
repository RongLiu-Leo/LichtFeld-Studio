/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_event_pool.hpp"

namespace lfs::core {

    CudaEventPool& CudaEventPool::instance() {
        static CudaEventPool pool;
        return pool;
    }

    cudaEvent_t CudaEventPool::acquire() {
        if (!shutdown_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!pool_.empty()) {
                cudaEvent_t event = pool_.back();
                pool_.pop_back();
                stats_.reused.fetch_add(1, std::memory_order_relaxed);
                return event;
            }
        }

        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            return nullptr;
        }
        stats_.created.fetch_add(1, std::memory_order_relaxed);
        return event;
    }

    void CudaEventPool::release(cudaEvent_t event) {
        if (!event)
            return;
        if (!shutdown_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pool_.size() < MAX_POOL_SIZE) {
                pool_.push_back(event);
                return;
            }
        }
        cudaEventDestroy(event);
    }

    void CudaEventPool::shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true))
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (cudaEvent_t event : pool_) {
            cudaEventDestroy(event);
        }
        pool_.clear();
    }

    CudaEventPool::~CudaEventPool() {
        shutdown();
    }

} // namespace lfs::core
