/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Shared scalar math for the LOD tree builders (bhatt_lod.cpp, octree_lod.cpp).
// Moved verbatim from bhatt_lod.cpp so both builders merge with bit-identical
// conventions (eigen decomposition, quaternion order, lodOpacity weights).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace lfs::core::lodmath {

    constexpr float kMinScale = 1e-12f;
    constexpr float kMinQuatNorm = 1e-12f;
    constexpr float kEllipsoidAreaP = 1.6075f;
    constexpr int kJacobiIterations = 10;
    constexpr float kMinEval = 1e-18f;
    constexpr float kEpsCov = 1e-8f;
    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float kPi = 3.14159265358979323846f;

    [[nodiscard]] inline float sigmoid(const float x) {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    }

    [[nodiscard]] inline float clamp_scale_raw(const float raw) {
        return std::clamp(raw, -30.0f, 30.0f);
    }

    [[nodiscard]] inline float activated_scale(const float raw) {
        return std::max(std::exp(clamp_scale_raw(raw)), kMinScale);
    }

    [[nodiscard]] inline float ellipsoid_area(const float sx, const float sy, const float sz) {
        const float t1 = std::pow(sx * sy, kEllipsoidAreaP);
        const float t2 = std::pow(sx * sz, kEllipsoidAreaP);
        const float t3 = std::pow(sy * sz, kEllipsoidAreaP);
        return 4.0f * kPi * std::pow((t1 + t2 + t3) / 3.0f, 1.0f / kEllipsoidAreaP);
    }

    [[nodiscard]] inline float lod_opacity(const float opacity) {
        if (opacity > 1.0f) {
            constexpr float kE = 2.718281828459045f;
            return std::sqrt(1.0f + kE * std::log(opacity));
        }
        return 1.0f;
    }

    inline void quat_to_rotmat(const float qw, const float qx, const float qy, const float qz, std::array<float, 9>& out) {
        const float xx = qx * qx;
        const float yy = qy * qy;
        const float zz = qz * qz;
        const float wx = qw * qx;
        const float wy = qw * qy;
        const float wz = qw * qz;
        const float xy = qx * qy;
        const float xz = qx * qz;
        const float yz = qy * qz;

        out[0] = 1.0f - 2.0f * (yy + zz);
        out[1] = 2.0f * (xy - wz);
        out[2] = 2.0f * (xz + wy);
        out[3] = 2.0f * (xy + wz);
        out[4] = 1.0f - 2.0f * (xx + zz);
        out[5] = 2.0f * (yz - wx);
        out[6] = 2.0f * (xz - wy);
        out[7] = 2.0f * (yz + wx);
        out[8] = 1.0f - 2.0f * (xx + yy);
    }

    inline void sigma_from_rot_var(const std::array<float, 9>& R,
                                   const float vx,
                                   const float vy,
                                   const float vz,
                                   std::array<float, 9>& out) {
        const std::array<float, 3> variance = {vx, vy, vz};
        std::array<float, 9> scaled{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                const size_t idx = static_cast<size_t>(row * 3 + col);
                scaled[idx] = R[idx] * variance[static_cast<size_t>(col)];
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                out[static_cast<size_t>(row * 3 + col)] =
                    scaled[static_cast<size_t>(row * 3 + 0)] * R[static_cast<size_t>(col * 3 + 0)] +
                    scaled[static_cast<size_t>(row * 3 + 1)] * R[static_cast<size_t>(col * 3 + 1)] +
                    scaled[static_cast<size_t>(row * 3 + 2)] * R[static_cast<size_t>(col * 3 + 2)];
            }
        }
    }

    [[nodiscard]] inline float det3(const std::array<float, 9>& A) {
        return A[0] * (A[4] * A[8] - A[5] * A[7]) -
               A[1] * (A[3] * A[8] - A[5] * A[6]) +
               A[2] * (A[3] * A[7] - A[4] * A[6]);
    }

    struct Eigen3x3 {
        std::array<float, 3> values{};
        std::array<float, 9> vectors{};
    };

    [[nodiscard]] inline Eigen3x3 sort_eigendecomposition(const Eigen3x3& out) {
        std::array<int, 3> order = {0, 1, 2};
        std::sort(order.begin(), order.end(), [&](const int lhs, const int rhs) {
            if (out.values[static_cast<size_t>(lhs)] != out.values[static_cast<size_t>(rhs)])
                return out.values[static_cast<size_t>(lhs)] > out.values[static_cast<size_t>(rhs)];
            return lhs < rhs;
        });

        Eigen3x3 sorted;
        for (int col = 0; col < 3; ++col) {
            const int src_col = order[static_cast<size_t>(col)];
            sorted.values[static_cast<size_t>(col)] = out.values[static_cast<size_t>(src_col)];
            for (int row = 0; row < 3; ++row)
                sorted.vectors[static_cast<size_t>(row * 3 + col)] = out.vectors[static_cast<size_t>(row * 3 + src_col)];
        }

        if (det3(sorted.vectors) < 0.0f) {
            sorted.vectors[2] *= -1.0f;
            sorted.vectors[5] *= -1.0f;
            sorted.vectors[8] *= -1.0f;
        }
        return sorted;
    }

    [[nodiscard]] inline Eigen3x3 eigen_symmetric_3x3_jacobi(const std::array<float, 9>& Ain) {
        std::array<float, 9> A = Ain;
        std::array<float, 9> V = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        };

        for (int iter = 0; iter < kJacobiIterations; ++iter) {
            int p = 0;
            int q = 1;
            float max_abs = std::abs(A[1]);
            if (std::abs(A[2]) > max_abs) {
                p = 0;
                q = 2;
                max_abs = std::abs(A[2]);
            }
            if (std::abs(A[5]) > max_abs) {
                p = 1;
                q = 2;
                max_abs = std::abs(A[5]);
            }
            if (max_abs < 1e-12f)
                break;

            const int pp = 3 * p + p;
            const int qq = 3 * q + q;
            const int pq = 3 * p + q;
            const float app = A[static_cast<size_t>(pp)];
            const float aqq = A[static_cast<size_t>(qq)];
            const float apq = A[static_cast<size_t>(pq)];
            const float tau = (aqq - app) / (2.0f * apq);
            const float t = std::copysign(1.0f, tau) / (std::abs(tau) + std::sqrt(1.0f + tau * tau));
            const float c = 1.0f / std::sqrt(1.0f + t * t);
            const float s = t * c;

            for (int k = 0; k < 3; ++k) {
                if (k == p || k == q)
                    continue;
                const int kp = 3 * k + p;
                const int kq = 3 * k + q;
                const float akp = A[static_cast<size_t>(kp)];
                const float akq = A[static_cast<size_t>(kq)];
                A[static_cast<size_t>(kp)] = c * akp - s * akq;
                A[static_cast<size_t>(3 * p + k)] = A[static_cast<size_t>(kp)];
                A[static_cast<size_t>(kq)] = s * akp + c * akq;
                A[static_cast<size_t>(3 * q + k)] = A[static_cast<size_t>(kq)];
            }

            A[static_cast<size_t>(pp)] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
            A[static_cast<size_t>(qq)] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
            A[static_cast<size_t>(pq)] = 0.0f;
            A[static_cast<size_t>(3 * q + p)] = 0.0f;

            for (int k = 0; k < 3; ++k) {
                const int kp = 3 * k + p;
                const int kq = 3 * k + q;
                const float vkp = V[static_cast<size_t>(kp)];
                const float vkq = V[static_cast<size_t>(kq)];
                V[static_cast<size_t>(kp)] = c * vkp - s * vkq;
                V[static_cast<size_t>(kq)] = s * vkp + c * vkq;
            }
        }

        Eigen3x3 out;
        out.values = {A[0], A[4], A[8]};
        out.vectors = V;
        return sort_eigendecomposition(out);
    }

    [[nodiscard]] inline Eigen3x3 eigen_symmetric_3x3(const std::array<float, 9>& Ain) {
        return eigen_symmetric_3x3_jacobi(Ain);
    }

    inline void rotmat_to_quat(const std::array<float, 9>& R, std::array<float, 4>& out) {
        const float m00 = R[0];
        const float m11 = R[4];
        const float m22 = R[8];
        const float tr = m00 + m11 + m22;
        float qw = 0.0f;
        float qx = 0.0f;
        float qy = 0.0f;
        float qz = 0.0f;

        if (tr > 0.0f) {
            const float S = std::sqrt(tr + 1.0f) * 2.0f;
            qw = 0.25f * S;
            qx = (R[7] - R[5]) / S;
            qy = (R[2] - R[6]) / S;
            qz = (R[3] - R[1]) / S;
        } else if (R[0] > R[4] && R[0] > R[8]) {
            const float S = std::sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;
            qw = (R[7] - R[5]) / S;
            qx = 0.25f * S;
            qy = (R[1] + R[3]) / S;
            qz = (R[2] + R[6]) / S;
        } else if (R[4] > R[8]) {
            const float S = std::sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;
            qw = (R[2] - R[6]) / S;
            qx = (R[1] + R[3]) / S;
            qy = 0.25f * S;
            qz = (R[5] + R[7]) / S;
        } else {
            const float S = std::sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;
            qw = (R[3] - R[1]) / S;
            qx = (R[2] + R[6]) / S;
            qy = (R[5] + R[7]) / S;
            qz = 0.25f * S;
        }

        const float inv_n = 1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
        out[0] = qw * inv_n;
        out[1] = qx * inv_n;
        out[2] = qy * inv_n;
        out[3] = qz * inv_n;
    }

    inline void decompose_sigma_to_raw_scale_quat(const std::array<float, 9>& sigma,
                                                  std::array<float, 3>& scaling_raw,
                                                  std::array<float, 4>& rotation_raw) {
        const auto eig = eigen_symmetric_3x3(sigma);
        std::array<float, 3> evals = {
            std::max(eig.values[0], kMinEval),
            std::max(eig.values[1], kMinEval),
            std::max(eig.values[2], kMinEval),
        };

        scaling_raw[0] = std::log(std::max(std::sqrt(evals[0]), kMinScale));
        scaling_raw[1] = std::log(std::max(std::sqrt(evals[1]), kMinScale));
        scaling_raw[2] = std::log(std::max(std::sqrt(evals[2]), kMinScale));
        rotmat_to_quat(eig.vectors, rotation_raw);
    }

    // Compute symmetric 3x3 covariance from scale + quaternion and store 6 unique elements + det
    inline void compute_covariance_from_scale_quat(
        float sx, float sy, float sz,
        float qw, float qx, float qy, float qz,
        float& out_xx, float& out_xy, float& out_xz,
        float& out_yy, float& out_yz, float& out_zz,
        float& out_det) {

        std::array<float, 9> R;
        quat_to_rotmat(qw, qx, qy, qz, R);
        const float sx2 = sx * sx;
        const float sy2 = sy * sy;
        const float sz2 = sz * sz;
        std::array<float, 9> cov;
        sigma_from_rot_var(R, sx2, sy2, sz2, cov);

        out_xx = cov[0];
        out_xy = cov[1];
        out_xz = cov[2];
        out_yy = cov[4];
        out_yz = cov[5];
        out_zz = cov[8];
        out_det = det3(cov);
    }

} // namespace lfs::core::lodmath
