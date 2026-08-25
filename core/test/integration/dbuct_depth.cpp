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
    // The grant reads the PARENT's visit count.  A backstep hands the child's
    // whole lump to its parent, so the two GAIN the same amount per camping
    // period; any head start the parent already had is preserved, not erased.
    // Here the tree starts empty and pos0 is on the path from the very first
    // episode, so it enters level with root and stays level, and both switch
    // from grant=1 to grant=2 on the same episode.  That is why no k camps pos0
    // alone in THIS fixture -- on a pre-seeded tree where root leads pos0 it
    // would.  (OOB is the counter-example: it joins one episode later and
    // trails by that one visit forever.)
    //
    // k=0.5, so grant = 1 + floor(0.5 * V):
    //   V=0,1 → grant 1: the child's budget is consumed by its single episode
    //           and the whole chain unwinds → depth 1.
    //   V=2   → grant 2 at both levels: pos0 and OOB each end the episode with
    //           lump 1 < budget 2, so both stay camped → depth 3.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);

    std::vector<int> path = {-1};

    // Each grant-1 episode banks exactly 1 visit everywhere on the path, so the
    // visit count walks 0 → 1 → 2 and the grant walks 1 → 1 → 2.

    // ep1: V=0 at choose, grant=1. Expand pos0, rollout from pos0.
    //      pos0 budget=1 exhausted immediately → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep2: V=1 at choose, grant=1 at both levels. Same pattern → depth=1.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep3: V=2 at choose, grant=2. pos0 gets budget=2 and hands OOB budget=2;
    //      OOB ends with lump=1<2, so nothing unwinds → depth=3, path={-1,0,1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 3u);
    EXPECT_EQ(path.back(), 1);

    // ep4: resumes at OOB (path={-1,0,1}), whose remaining budget is 1, so its
    //      child gets grant 1 and exhausts.  That fills OOB's lump to 2 and
    //      pos0's to 2, so the whole chain unwinds → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);
}

TEST_F(DbuctDepthTest, ManualBackstepToRootOverridesCamping)
{
    // Same k=0.5 setup as above.  Episode 3 would normally leave both pos0 and
    // OOB camped (depth=3).  Caller invokes backstep() after terminate() to
    // climb back to root, and each partial lump is rolled into its parent as a
    // budget-driven backstep would do.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);

    std::vector<int> path = {-1};

    // Advance through eps 1-2 (V=0,1 → grant=1; camping begins at V=2).
    run_episode(m, track, jumps, path);
    run_episode(m, track, jumps, path);
    ASSERT_EQ(path.back(), -1);

    const size_t root_visits_before = visits.get_visits(-1);

    // ep3: grant=2, the chain would camp at depth=3 — caller backsteps to root.
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

    // Verify that the single episode's visit reached root through both partial
    // lumps, even though neither budget was naturally exhausted.
    EXPECT_EQ(visits.get_visits(-1) - root_visits_before, 1u);
}
