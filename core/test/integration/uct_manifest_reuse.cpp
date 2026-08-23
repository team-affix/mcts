// A manifest driven for many episodes must produce exactly the stats of a fresh
// manifest per episode: terminate() has to leave the cursor and path
// indistinguishable from construction.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class UctManifestReuseTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::uct_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;

    // Terminal-reward episode: reward is the last in-bounds track value.
    void run_episode(manifest_t&                m,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps)
    {
        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                return;
            }
            position = next;
            reward   = track[position];
        }
    }
};

TEST_F(UctManifestReuseTest, ReusedManifestMatchesFreshManifestPerEpisode)
{
    const std::vector<double> track = {3.0, 1.0, 4.0, 1.0, 5.0};
    const std::vector<jump_t> jumps = {1, 2};
    const double              c     = 1.4;
    const int                 episodes = 40;

    visits_t     reused_visits, fresh_visits;
    value_t      reused_value,  fresh_value;
    std::mt19937 reused_rng(7), fresh_rng(7);

    {
        manifest_t m(reused_visits, reused_visits, reused_value, reused_value,
                     reused_rng, c, -1);

        for (int i = 0; i < episodes; ++i)
            run_episode(m, track, jumps);
    }

    for (int i = 0; i < episodes; ++i)
    {
        manifest_t m(fresh_visits, fresh_visits, fresh_value, fresh_value,
                     fresh_rng, c, -1);
        run_episode(m, track, jumps);
    }

    // Covers the root (-1), every track position, and the out-of-bounds
    // handles that land on the backprop path before termination.
    for (int pos = -1; pos <= static_cast<int>(track.size()) + 2; ++pos)
    {
        EXPECT_EQ(reused_visits.get_visits(pos), fresh_visits.get_visits(pos))
            << "visit mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(reused_value.get_value(pos), fresh_value.get_value(pos))
            << "value mismatch at pos=" << pos;
    }

    EXPECT_GT(reused_visits.get_visits(-1), 0u);
}
