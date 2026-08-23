// Verifies the in_rollout flag through the real uct manifest across two episodes
// driven by a single manifest, which only works because terminate() restores the
// cursor and backprop path.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class UctInRolloutTest : public ::testing::Test
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
};

TEST_F(UctInRolloutTest, FlagTransitionsEpisodes1And2)
{
    const std::vector<double> track = {5.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    // One manifest drives both episodes: terminate() restores the cursor and path.
    manifest_t m(visits, visits, value, value, rng, 1.0, -1);

    // Episode 1: root has 0 visits, so the flag flips at the first expansion.
    EXPECT_FALSE(m.in_rollout.is_in_rollout());

    m.chooser.choose(jumps, jumps);
    EXPECT_TRUE(m.in_rollout.is_in_rollout());

    m.chooser.choose(jumps, jumps);
    EXPECT_TRUE(m.in_rollout.is_in_rollout());

    m.delta.set_value(5.0);
    m.terminator.terminate();
    EXPECT_FALSE(m.in_rollout.is_in_rollout());

    // Episode 2: root now has a visit, so the first choose() stays in the tree.
    m.chooser.choose(jumps, jumps);
    EXPECT_FALSE(m.in_rollout.is_in_rollout());

    m.chooser.choose(jumps, jumps);
    EXPECT_TRUE(m.in_rollout.is_in_rollout());

    m.delta.set_value(5.0);
    m.terminator.terminate();
    EXPECT_FALSE(m.in_rollout.is_in_rollout());
}
