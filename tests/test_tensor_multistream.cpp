/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_event_pool.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"

using namespace lfs::core;

namespace {

    constexpr size_t SLAB_BYTES = 64 * 1024;
    constexpr size_t BUCKET_BYTES = 4 * 1024 * 1024;

    void destroyStreamSafely(cudaStream_t stream) {
        CudaMemoryPool::instance().release_stream(stream);
        cudaStreamDestroy(stream);
    }

    class GateStream {
    public:
        GateStream() {
            EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
            EXPECT_EQ(cudaEventCreateWithFlags(&gate_, cudaEventDisableTiming), cudaSuccess);
            EXPECT_EQ(cudaStreamCreateWithFlags(&gate_holder_, cudaStreamNonBlocking), cudaSuccess);
        }

        ~GateStream() {
            release();
            destroyStreamSafely(stream_);
            cudaStreamDestroy(gate_holder_);
            cudaEventDestroy(gate_);
        }

        // Blocks `stream_` behind a host-controlled gate so work enqueued after
        // close() cannot run until release() — widens race windows deterministically.
        void close() {
            EXPECT_EQ(cudaLaunchHostFunc(
                          gate_holder_,
                          [](void* userData) {
                              auto* released = static_cast<std::atomic<bool>*>(userData);
                              while (!released->load(std::memory_order_acquire)) {
                              }
                          },
                          &released_),
                      cudaSuccess);
            EXPECT_EQ(cudaEventRecord(gate_, gate_holder_), cudaSuccess);
            EXPECT_EQ(cudaStreamWaitEvent(stream_, gate_, 0), cudaSuccess);
        }

        void release() {
            released_.store(true, std::memory_order_release);
        }

        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
        cudaStream_t gate_holder_ = nullptr;
        cudaEvent_t gate_ = nullptr;
        std::atomic<bool> released_{false};
    };

} // namespace

class TensorMultiStreamTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};

TEST_F(TensorMultiStreamTest, SlabSameStreamReuseIsImmediateAndStealFree) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();
    const uint64_t steals_before = GPUSlabAllocator::instance().stats().steal_count.load();

    void* first = pool.allocate(SLAB_BYTES, stream);
    ASSERT_NE(first, nullptr);
    pool.deallocate(first, stream);

    void* second = pool.allocate(SLAB_BYTES, stream);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second, first);
    EXPECT_EQ(GPUSlabAllocator::instance().stats().steal_count.load(), steals_before);

    pool.deallocate(second, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    destroyStreamSafely(stream);
}

TEST_F(TensorMultiStreamTest, SlabCrossStreamStealIsOrdered) {
    GateStream producer;
    cudaStream_t consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();

    void* block = pool.allocate(SLAB_BYTES, producer.get());
    ASSERT_NE(block, nullptr);

    producer.close();
    ASSERT_EQ(cudaMemsetAsync(block, 0xAA, SLAB_BYTES, producer.get()), cudaSuccess);
    pool.deallocate(block, producer.get());

    const uint64_t steals_before = GPUSlabAllocator::instance().stats().steal_count.load();
    void* stolen = pool.allocate(SLAB_BYTES, consumer);
    ASSERT_NE(stolen, nullptr);

    if (stolen == block) {
        EXPECT_GT(GPUSlabAllocator::instance().stats().steal_count.load(), steals_before);
        ASSERT_EQ(cudaMemsetAsync(stolen, 0xBB, SLAB_BYTES, consumer), cudaSuccess);

        producer.release();
        ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);

        std::vector<unsigned char> host(SLAB_BYTES);
        ASSERT_EQ(cudaMemcpy(host.data(), stolen, SLAB_BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < SLAB_BYTES; i += 4096) {
            ASSERT_EQ(host[i], 0xBB) << "producer write ordered after steal at offset " << i;
        }
    } else {
        producer.release();
    }

    pool.deallocate(stolen, consumer);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    destroyStreamSafely(consumer);
}

TEST_F(TensorMultiStreamTest, BucketCrossStreamReuseIsOrdered) {
    GateStream producer;
    cudaStream_t consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();

    void* block = pool.allocate(BUCKET_BYTES, producer.get());
    ASSERT_NE(block, nullptr);

    producer.close();
    ASSERT_EQ(cudaMemsetAsync(block, 0xAA, BUCKET_BYTES, producer.get()), cudaSuccess);
    pool.deallocate(block, producer.get());

    const uint64_t cross_before = SizeBucketedPool::instance().stats().cross_stream_reuse.load();
    void* reused = pool.allocate(BUCKET_BYTES, consumer);
    ASSERT_NE(reused, nullptr);

    if (reused == block) {
        EXPECT_GT(SizeBucketedPool::instance().stats().cross_stream_reuse.load(), cross_before);
        ASSERT_EQ(cudaMemsetAsync(reused, 0xBB, BUCKET_BYTES, consumer), cudaSuccess);

        producer.release();
        ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);

        std::vector<unsigned char> host(BUCKET_BYTES);
        ASSERT_EQ(cudaMemcpy(host.data(), reused, BUCKET_BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < BUCKET_BYTES; i += 65536) {
            ASSERT_EQ(host[i], 0xBB) << "producer write ordered after reuse at offset " << i;
        }
    } else {
        producer.release();
    }

    pool.deallocate(reused, consumer);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    destroyStreamSafely(consumer);
}

TEST_F(TensorMultiStreamTest, RecordStreamBridgesReaderIntoFree) {
    GateStream reader;
    cudaStream_t owner;
    ASSERT_EQ(cudaStreamCreateWithFlags(&owner, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();

    void* block = pool.allocate(SLAB_BYTES, owner);
    ASSERT_NE(block, nullptr);
    ASSERT_EQ(cudaMemsetAsync(block, 0xCC, SLAB_BYTES, owner), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(owner), cudaSuccess);

    std::vector<unsigned char> readback(SLAB_BYTES, 0);
    void* staging = nullptr;
    ASSERT_EQ(cudaMalloc(&staging, SLAB_BYTES), cudaSuccess);

    reader.close();
    ASSERT_EQ(cudaMemcpyAsync(staging, block, SLAB_BYTES, cudaMemcpyDeviceToDevice, reader.get()),
              cudaSuccess);
    pool.record_stream(block, reader.get());
    pool.deallocate(block, owner);

    void* reused = pool.allocate(SLAB_BYTES, owner);
    if (reused == block) {
        ASSERT_EQ(cudaMemsetAsync(reused, 0xDD, SLAB_BYTES, owner), cudaSuccess);
    }

    reader.release();
    ASSERT_EQ(cudaStreamSynchronize(reader.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(owner), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(readback.data(), staging, SLAB_BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < SLAB_BYTES; i += 4096) {
        ASSERT_EQ(readback[i], 0xCC) << "reader saw overwrite from recycled block at offset " << i;
    }

    if (reused) {
        pool.deallocate(reused, owner);
    }
    cudaFree(staging);
    destroyStreamSafely(owner);
}

TEST_F(TensorMultiStreamTest, MultiThreadMultiStreamHammer) {
    constexpr int kThreads = 4;
    constexpr int kIterations = 12000;
    static constexpr size_t kSizes[] = {1024, 96 * 1024, 512 * 1024, 3 * 1024 * 1024};

    std::vector<std::thread> threads;
    std::atomic<int> failures{0};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t, &failures] {
            cudaSetDevice(0);
            cudaStream_t stream;
            if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
                failures.fetch_add(1);
                return;
            }
            auto& pool = CudaMemoryPool::instance();
            const unsigned char pattern = static_cast<unsigned char>(0x10 + t);

            for (int i = 0; i < kIterations; ++i) {
                const size_t bytes = kSizes[(t + i) % 4];
                void* ptr = pool.allocate(bytes, stream);
                if (!ptr) {
                    failures.fetch_add(1);
                    break;
                }
                if (cudaMemsetAsync(ptr, pattern, bytes, stream) != cudaSuccess) {
                    failures.fetch_add(1);
                    pool.deallocate(ptr, stream);
                    break;
                }
                if (i % 512 == 0) {
                    unsigned char probe = 0;
                    if (cudaMemcpyAsync(&probe, ptr, 1, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                        cudaStreamSynchronize(stream) != cudaSuccess || probe != pattern) {
                        failures.fetch_add(1);
                        pool.deallocate(ptr, stream);
                        break;
                    }
                }
                pool.deallocate(ptr, stream);
            }

            cudaStreamSynchronize(stream);
            destroyStreamSafely(stream);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(failures.load(), 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
