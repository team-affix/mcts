// Verifies grant_k = 1 + floor(k * visits(child)) by observing the delta in
// bank.get_visits(root) after each camping grant period completes.  Root's visit
// count only increases when a child frame backsteps to root, and increases by
// exactly grant_k (the child's budget), making the jump in bank.get_visits(-1)
// the sole public observable needed.
//
// The grant reads pos0, the child root hands the budget to.  Root banks exactly
// the lump pos0 accumulated, so the two counts coincide at every period boundary
// -- which is the instant root's choose() computes the grant -- and diverge only
// mid-period, while pos0 runs ahead.  The per-period grants therefore follow a
// closed-form recurrence in the shared boundary value:
//     V_next = V + 1 + floor(k * V), starting from V = 1 after the seed episode.
// The expectations below are that recurrence unrolled by hand, not fitted to
// observed output.

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

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithChildVisitsKHalf)
{
    // Single-path game: root -> pos0 -> OOB.  Only the grant root hands to pos0 is
    // under test, and at period start pos0's count equals root's, so
    // grant_k = 1 + floor(0.5 * V) with V the count observed at period start.
    //
    // V:      1  2  3  4  6  9 14 21 31 47   <- grant paid out this period
    // V_next: 2  4  7 11 17 26 40 61 92 139
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const std::vector<size_t> expected_grants = {1, 2, 3, 4, 6, 9, 14, 21, 31, 47};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.5, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit; the grant here is 1 + floor(k * 0) = 1.
    run_grant_period(m, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    for (size_t period = 0; period < expected_grants.size(); ++period)
    {
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(visits.get_visits(-1) - V_before, expected_grants[period])
            << "grant mismatch at period=" << period << " V_before=" << V_before;
    }

    EXPECT_EQ(visits.get_visits(-1), 139u);
}

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithChildVisitsKQuarter)
{
    // Same game, k = 0.25: the grant stays at 1 while floor(0.25 * V) is 0,
    // i.e. for V = 1..3, then climbs.
    //
    // V:      1  1  1  2  2  3  3  4  5  6  8 10
    // V_next: 2  3  4  6  8 11 14 18 23 29 37 47
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const std::vector<size_t> expected_grants = {1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 8, 10};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.25, -1);
    std::vector<int> path = {-1};

    run_grant_period(m, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    for (size_t period = 0; period < expected_grants.size(); ++period)
    {
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(visits.get_visits(-1) - V_before, expected_grants[period])
            << "grant mismatch at period=" << period << " V_before=" << V_before;
    }

    EXPECT_EQ(visits.get_visits(-1), 47u);
}

TEST_F(DbuctGrantFormulaTest, ZeroKGrantsOneForever)
{
    // k = 0 is the vanilla-UCT setting: every period is a single episode.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 0.0, -1);
    std::vector<int> path = {-1};

    for (size_t period = 0; period < 20; ++period)
    {
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(visits.get_visits(-1) - V_before, 1u)
            << "grant mismatch at period=" << period;
    }

    EXPECT_EQ(visits.get_visits(-1), 20u);
}
