// Verifies the in_rollout flag through the real dbuct manifest: false before any
// choose() in an episode, flips to true exactly when the expansion node is
// encountered, and resets to false after terminate().

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctInRolloutTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::dbuct_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;
};

TEST_F(DbuctInRolloutTest, FlagTransitionsEpisodes1And2)
{
    // track={5.0}: positions root(-1), pos0(0), OOB at 1.
    // Every episode uses exactly two choose() calls before terminate().
    const std::vector<double> track = {5.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 1.0, 0.0, -1);

    // Episode 1: root has 0 visits → in_rollout flips on the very first choose().
    EXPECT_FALSE(m.in_rollout.is_in_rollout());

    m.chooser.choose(jumps, jumps);        // at root (0 visits): immediate rollout, no frame pushed
    EXPECT_TRUE(m.in_rollout.is_in_rollout());   // flipped at expansion (root itself is expansion node)

    m.chooser.choose(jumps, jumps);        // still in rollout (pos0 → jump → OOB next)
    EXPECT_TRUE(m.in_rollout.is_in_rollout());   // flag persists until terminate()

    m.delta.set_value(5.0);
    m.terminator.terminate();   // resets flag
    EXPECT_FALSE(m.in_rollout.is_in_rollout());

    // Episode 2: root (1 visit) → UCB → pos0 pushed; pos0 (0 visits) → expansion.
    EXPECT_FALSE(m.in_rollout.is_in_rollout());

    m.chooser.choose(jumps, jumps);        // UCB at root (1 visit): tree phase, pos0 frame pushed
    EXPECT_FALSE(m.in_rollout.is_in_rollout()); // still tree phase — flag not set during UCB selection

    m.chooser.choose(jumps, jumps);        // at pos0 (0 visits): expansion, rollout chosen
    EXPECT_TRUE(m.in_rollout.is_in_rollout());  // flipped exactly here

    m.delta.set_value(5.0);
    m.terminator.terminate();
    EXPECT_FALSE(m.in_rollout.is_in_rollout());
}
