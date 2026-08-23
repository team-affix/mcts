// terminate() must credit every node the episode recorded on the backprop path,
// not just the leaf.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class UctTerminateBackpropTest : public ::testing::Test
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

    void run_terminal_episode(manifest_t& m, const std::vector<jump_t>& jumps)
    {
        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= 3)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                return;
            }
            position = next;
            reward   = static_cast<double>(next + 1);
        }
    }
};

TEST_F(UctTerminateBackpropTest, CreditsEveryNodeOnBackpropPath)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, -1);

    run_terminal_episode(m, jumps);

    EXPECT_EQ(visits.get_visits(-1), 1u);
    EXPECT_EQ(visits.get_visits(0), 1u);
    EXPECT_DOUBLE_EQ(value.get_value(-1), 3.0);
    EXPECT_DOUBLE_EQ(value.get_value(0), 3.0);
}
