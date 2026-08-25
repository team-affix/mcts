// Verifies the frame stack size after terminate() reflects budget-driven
// backtracking.  Root only = 1; every unexhausted frame left on the stack adds
// one level.  Callers sync their path via path.resize(frame_stack.size()).

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
                          std::mt19937>;

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
    // Single-step game: root(-1) → pos0(0) → OOB(1).
    //
    // The grant reads the CHILD's visit count, so each level is sized
    // independently: grant = 1 + floor(0.5 * visits(child)).  OOB joins the tree
    // one episode after pos0 and trails it by that one visit forever, which is
    // what lets pos0 camp while OOB does not.
    //
    //   pos0 at 0 or 1 visits → grant 1: its single episode consumes the budget
    //        and the whole chain unwinds → depth 1.
    //   pos0 at 2 visits → grant 2, while OOB (still at 1) only earns grant 1.
    //        OOB exhausts and rolls into pos0, leaving pos0 with lump 1 < 2 →
    //        pos0 alone stays camped → depth 2.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);

    std::vector<int> path = {-1};

    // Each grant-1 episode banks 1 visit on every node it passes through, so
    // pos0 walks 0 → 1 → 2 while OOB walks 0 → 1.

    // ep1: pos0 at 0 visits, grant=1. Expand pos0, rollout from pos0.
    //      pos0 budget=1 exhausted immediately → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep2: pos0 at 1 visit, grant=1; OOB is fresh, so it too gets grant=1.
    //      Both exhaust → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep3: pos0 at 2 visits → budget=2; OOB at 1 visit → budget=1.  OOB exhausts
    //      and rolls its lump into pos0, whose lump is then 1 < 2 → pos0 camps,
    //      depth=2, path={-1,0}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 2u);
    EXPECT_EQ(path.back(), 0);

    // ep4: resumes at pos0 (path={-1,0}) with 1 of its 2 budget spent.  Its
    //      second episode fills the lump → backstep to root, depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);
}

TEST_F(DbuctDepthTest, ManualBackstepToRootOverridesCamping)
{
    // Same k=0.5 setup as above.  Episode 3 would normally leave pos0 camped
    // (depth=2).  Caller invokes backstep() after terminate() to climb back to
    // root, and pos0's partial lump is rolled into root as a budget-driven
    // backstep would do.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);

    std::vector<int> path = {-1};

    // Advance through eps 1-2 (pos0 at 0,1 visits → grant=1; camping begins at 2).
    run_episode(m, track, jumps, path);
    run_episode(m, track, jumps, path);
    ASSERT_EQ(path.back(), -1);

    const size_t root_visits_before = visits.get_visits(-1);

    // ep3: pos0 gets budget=2 and would camp at depth=2 — caller backsteps out.
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

    // Verify that pos0's partial lump (1 visit) was rolled into root even though
    // pos0's budget was not naturally exhausted.
    EXPECT_EQ(visits.get_visits(-1) - root_visits_before, 1u);
}
