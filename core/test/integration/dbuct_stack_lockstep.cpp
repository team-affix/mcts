// The frame stack and the value stack are pushed and popped by separate
// controllers, so their tops must stay on the same handle throughout an episode.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctStackLockstepTest : public ::testing::Test
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

    void run_episode(manifest_t& m, const std::vector<jump_t>& jumps)
    {
        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.is_in_rollout())
                EXPECT_EQ(m.frame_stack.top().handle, m.value_stack.top().handle);
            if (next >= 1)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                return;
            }
            position = next;
            reward   = 1.0;
        }
    }
};

TEST_F(DbuctStackLockstepTest, FrameAndValueStacksShareTopHandle)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);

    // With k=0.5 the first two episodes run at grant 1 and unwind completely.  On
    // the third, pos0 has 2 visits and earns budget 2 while the OOB node still has
    // 1 and earns budget 1, so only pos0 stays camped -- and that is where the two
    // stacks must still agree.
    run_episode(m, jumps);
    run_episode(m, jumps);
    run_episode(m, jumps);

    EXPECT_EQ(m.frame_stack.size(), 2u);
    EXPECT_EQ(m.frame_stack.top().handle, 0);
    EXPECT_EQ(m.value_stack.top().handle, 0);
}
