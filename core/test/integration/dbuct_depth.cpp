// Verifies the frame stack size after terminate() reflects budget-driven
// backtracking.  Root only = 1; a child camping one level deep = 2.  Callers
// sync their path via path.resize(frame_stack.size()).

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctDepthTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::dbuct_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937, std::unordered_map>;

    void run_episode(manifest_t&                m,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.is_in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                path.resize(m.frame_stack.size());
                return;
            }
            position = next;
            reward   = track[position];
        }
    }
};

TEST_F(DbuctDepthTest, ReturnsCorrectCampingFrameIndex)
{
    // Single-step game: root(-1) → pos0 → OOB.
    // GII=2: dispatches D=0,1 give grant=1 (budget=1, always fully consumed → depth=1).
    //        dispatch  D=2     gives grant=2 (budget=2 at pos0):
    //          first  episode under that budget: pos0 not exhausted → camping, depth=2.
    //          second episode under that budget: pos0 exhausted     → backstep to root, depth=1.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 2, -1);   // GII = 2

    std::vector<int> path = {-1};

    // Expansion-first: choose() always dispatches a child before rolling out,
    // so every episode increments the dispatch counter.
    // D=0 → grant=1, D=1 → grant=1, D=2 → grant=2 (camping begins).

    // ep1: D(-1)=0 before dispatch, grant=1. Expand pos0, rollout from pos0.
    //      pos0 budget=1 exhausted immediately → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep2: D(-1)=1 before dispatch, grant=1. Same pattern → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep3: D(-1)=2 before dispatch, grant=2. pos0 gets budget=2; after 1 sim
    //      visit_lump=1<2 → camping at pos0, depth=2, path={-1, 0}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 2u);
    EXPECT_EQ(path.back(), 0);

    // ep4: continuing from pos0 (path={-1,0}). pos0's second sim exhausts budget=2
    //      → backstep to root, depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);
}

TEST_F(DbuctDepthTest, ManualBackstepToRootOverridesCamping)
{
    // Same GII=2 setup as above.  Episode 4 would normally camp at pos0
    // (depth=2).  Caller invokes backstep() after terminate() to climb to root,
    // and the pos0 lump is rolled into root's lump as usual.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 2, -1);   // GII = 2

    std::vector<int> path = {-1};

    // Advance through eps 1-2 (D=0,1 → grant=1 periods; camping begins at D=2).
    run_episode(m, track, jumps, path);
    run_episode(m, track, jumps, path);
    ASSERT_EQ(path.back(), -1);

    const size_t root_visits_before = visits.get_visits(-1);

    // ep4: grant=2, pos0 would camp at depth=2 — caller backsteps to root.
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.is_in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                while (m.frame_stack.size() > 1)
                    m.value_stack_controller.backstep();
                path.resize(m.frame_stack.size());
                break;
            }
            position = next;
            reward   = track[position];
        }
    }

    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // Verify that pos0's partial lump (1 visit) was rolled into root even
    // though pos0's budget was not naturally exhausted.
    EXPECT_EQ(visits.get_visits(-1) - root_visits_before, 1u);
}
