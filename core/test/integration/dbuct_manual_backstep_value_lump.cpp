// A caller-driven backstep after terminate() must roll the camping node's
// accumulated value lump into its parent, exactly as a budget-driven backstep
// would.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctManualBackstepValueLumpTest : public ::testing::Test
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

    void run_episode(manifest_t& m, const std::vector<jump_t>& jumps, std::vector<int>& path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.is_in_rollout())
                path.push_back(next);
            if (next >= 1)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                path.resize(m.frame_stack.size());
                return;
            }
            position = next;
            reward   = 1.0;
        }
    }
};

TEST_F(DbuctManualBackstepValueLumpTest, ManualBackstepRollsValueLumpIntoRoot)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);
    std::vector<int> path = {-1};

    run_episode(m, jumps, path);
    run_episode(m, jumps, path);

    const double root_value_before = value.get_value(-1);

    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.is_in_rollout())
                path.push_back(next);
            if (next >= 1)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                while (m.frame_stack.size() > 1)
                    m.value_stack_controller.backstep();
                path.resize(m.frame_stack.size());
                break;
            }
            position = next;
            reward   = 1.0;
        }
    }

    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_DOUBLE_EQ(value.get_value(-1) - root_value_before, 1.0);
}
