// Verifies grant_k = 1 + N / GII for several N values by observing the delta in
// bank.get_visits(root) after each camping grant period completes.  Root's visit
// count only increases when a child frame backsteps to root, and increases by
// exactly grant_k (the child's budget), making the jump in bank.get_visits(-1)
// the sole public observable needed.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctGrantFormulaTest : public ::testing::Test
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

    // Runs episodes until visits.get_visits(-1) increases, then returns the delta.
    // The delta equals the grant_k assigned to the child that just backstep-ed.
    size_t run_grant_period(manifest_t&                m,
                            visits_t&                  visits,
                            const std::vector<double>& track,
                            const std::vector<jump_t>& jumps,
                            std::vector<int>&          path)
    {
        size_t before = visits.get_visits(-1);
        while (visits.get_visits(-1) == before)
            run_episode(m, track, jumps, path);
        return visits.get_visits(-1) - before;
    }
};

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithRootDispatchesGII3)
{
    // Single-path game: root → pos0 → OOB.
    // Each grant period root dispatches pos0 exactly once (D increments by 1).
    // grant_k = 1 + D_before / GII, verified via the delta in visits.get_visits(-1).
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 3;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, GII, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit (rollout phase; no UCB dispatch happens here).
    run_grant_period(m, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    // From here, every period involves exactly one UCB dispatch from root.
    // Formula: grant_k = 1 + D_before / GII (integer division).
    for (size_t period = 0; period < 10; ++period)
    {
        const size_t D_before = m.dispatches.get_dispatches(-1);
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(m.dispatches.get_dispatches(-1), D_before + 1)
            << "dispatch count did not increment at period=" << period;
        EXPECT_EQ(visits.get_visits(-1) - V_before, 1 + D_before / GII)
            << "grant mismatch at period=" << period
            << " D_before=" << D_before << " GII=" << GII;
    }
}

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithRootDispatchesGII5)
{
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 5;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, GII, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit (rollout phase; no UCB dispatch happens here).
    run_grant_period(m, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    // GII=5: grant stays 1 for D=0..4, then rises by 1 every 5 dispatches.
    for (size_t period = 0; period < 12; ++period)
    {
        const size_t D_before = m.dispatches.get_dispatches(-1);
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(m.dispatches.get_dispatches(-1), D_before + 1)
            << "dispatch count did not increment at period=" << period;
        EXPECT_EQ(visits.get_visits(-1) - V_before, 1 + D_before / GII)
            << "grant mismatch at period=" << period
            << " D_before=" << D_before << " GII=" << GII;
    }
}
