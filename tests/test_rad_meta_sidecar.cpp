/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "io/formats/rad.hpp"
#include "io/ply_to_rad_lod.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <vector>

namespace {

    using lfs::core::NodeLinksRecord;

    NodeLinksRecord makeNode(const std::uint32_t child_start, const std::uint32_t child_count) {
        return {
            .child_start = child_start,
            .packed = child_count & 0xffffu,
            .parent = 0xFFFFFFFFu,
            .logical = 0,
        };
    }

    TEST(RadMetaSidecar, DeriveParentsHandlesNonMonotoneChildStart) {
        // Level-ordered tree whose level-1 parents point at non-monotone
        // child_start ranges, as the multi-bucket converter layouts produce:
        // node 0 (root) -> [1, 3); node 1 -> [5, 7); node 2 -> [3, 5).
        std::vector<NodeLinksRecord> links{
            makeNode(1, 2),
            makeNode(5, 2),
            makeNode(3, 2),
            makeNode(0, 0),
            makeNode(0, 0),
            makeNode(0, 0),
            makeNode(0, 0),
        };
        const auto leaf_count = lfs::io::derive_rad_meta_parents_levels(std::span(links));
        ASSERT_TRUE(leaf_count.has_value()) << leaf_count.error();
        EXPECT_EQ(*leaf_count, 4u);

        EXPECT_EQ(links[0].parent, 0xFFFFFFFFu);
        EXPECT_EQ(links[1].parent, 0u);
        EXPECT_EQ(links[2].parent, 0u);
        EXPECT_EQ(links[3].parent, 2u);
        EXPECT_EQ(links[4].parent, 2u);
        EXPECT_EQ(links[5].parent, 1u);
        EXPECT_EQ(links[6].parent, 1u);
        EXPECT_EQ(links[0].level(), 0u);
        EXPECT_EQ(links[1].level(), 1u);
        EXPECT_EQ(links[2].level(), 1u);
        for (std::size_t i = 3; i < links.size(); ++i) {
            EXPECT_EQ(links[i].level(), 2u);
        }
    }

    TEST(RadMetaSidecar, DeriveParentsRejectsCorruptLayouts) {
        // Child range pointing backwards.
        std::vector<NodeLinksRecord> backwards{makeNode(0, 1), makeNode(0, 0)};
        EXPECT_FALSE(lfs::io::derive_rad_meta_parents_levels(std::span(backwards)).has_value());

        // Orphan node (no parent assigns it).
        std::vector<NodeLinksRecord> orphan{makeNode(1, 1), makeNode(0, 0), makeNode(0, 0)};
        EXPECT_FALSE(lfs::io::derive_rad_meta_parents_levels(std::span(orphan)).has_value());
    }

    struct SyntheticSplat {
        float x, y, z;
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        float dc0, dc1, dc2;
        float opacity;
        float s0, s1, s2;
        float r0, r1, r2, r3;
    };

    void writeSyntheticPly(const std::filesystem::path& path, const std::size_t count) {
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> pos(-50.0f, 50.0f);
        std::vector<SyntheticSplat> splats(count);
        for (auto& s : splats) {
            s = {.x = pos(rng), .y = pos(rng), .z = pos(rng), .dc0 = 0.1f, .dc1 = 0.2f, .dc2 = 0.3f, .opacity = 1.0f, .s0 = -4.0f, .s1 = -4.0f, .s2 = -4.0f, .r0 = 1.0f, .r1 = 0.0f, .r2 = 0.0f, .r3 = 0.0f};
        }
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good());
        out << "ply\nformat binary_little_endian 1.0\n"
            << "element vertex " << splats.size() << "\n";
        for (const char* name : {"x", "y", "z", "nx", "ny", "nz",
                                 "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                 "scale_0", "scale_1", "scale_2",
                                 "rot_0", "rot_1", "rot_2", "rot_3"}) {
            out << "property float " << name << "\n";
        }
        out << "end_header\n";
        static_assert(sizeof(SyntheticSplat) == 17 * sizeof(float));
        out.write(reinterpret_cast<const char*>(splats.data()),
                  static_cast<std::streamsize>(splats.size() * sizeof(SyntheticSplat)));
        ASSERT_TRUE(out.good());
    }

    std::filesystem::path makeTestRad(const std::filesystem::path& temp_dir) {
        std::filesystem::create_directories(temp_dir);
        const auto ply_path = temp_dir / "sidecar.ply";
        const auto rad_path = temp_dir / "sidecar.rad";
        writeSyntheticPly(ply_path, 200'000);
        lfs::io::PlyToRadLodOptions options;
        options.target_bucket_splats = 65'536;
        options.temp_dir = temp_dir / "scratch";
        EXPECT_TRUE(lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options).has_value());
        return rad_path;
    }

    TEST(RadMetaSidecar, RoundtripAndInvalidation) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_meta_sidecar";
        std::filesystem::remove_all(temp_dir);
        const auto rad_path = makeTestRad(temp_dir);

        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());
        const auto meta_path = lfs::io::rad_meta_sidecar_path(rad_path);
        ASSERT_TRUE(std::filesystem::exists(meta_path));

        auto view = lfs::io::open_rad_meta_sidecar(rad_path);
        ASSERT_TRUE(view.has_value()) << view.error();
        EXPECT_GT(view->node_count, 200'000u);
        EXPECT_EQ(view->leaf_count, 200'000u);
        EXPECT_EQ(view->links[0].parent, 0xFFFFFFFFu);
        EXPECT_EQ(view->links[0].logical, 0u);

        // Touching the RAD file invalidates the sidecar (mtime/hash stamp).
        {
            std::ofstream touch(rad_path, std::ios::binary | std::ios::app);
            touch.put('\0');
        }
        auto stale = lfs::io::open_rad_meta_sidecar(rad_path);
        EXPECT_FALSE(stale.has_value());

        // A rebuild repairs it.
        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(rad_path).has_value());
        EXPECT_TRUE(lfs::io::open_rad_meta_sidecar(rad_path).has_value());

        // An incomplete build (complete flag cleared) is rejected.
        {
            std::fstream corrupt(meta_path, std::ios::binary | std::ios::in | std::ios::out);
            corrupt.seekp(89, std::ios::beg); // RadMetaHeader::complete byte offset
            corrupt.put('\0');
        }
        auto incomplete = lfs::io::open_rad_meta_sidecar(rad_path);
        EXPECT_FALSE(incomplete.has_value());

        std::filesystem::remove_all(temp_dir);
    }

#ifndef _WIN32
    TEST(RadMetaSidecar, BuildFailsCleanlyInReadOnlyDirectory) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "rad_meta_sidecar_ro";
        std::filesystem::remove_all(temp_dir);
        const auto rad_path = makeTestRad(temp_dir);

        std::filesystem::permissions(temp_dir,
                                     std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace);
        const auto result = lfs::io::build_rad_meta_sidecar(rad_path);
        std::filesystem::permissions(temp_dir,
                                     std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace);

        ASSERT_FALSE(result.has_value());
        EXPECT_FALSE(result.error().message.empty());

        std::filesystem::remove_all(temp_dir);
    }
#endif

} // namespace
