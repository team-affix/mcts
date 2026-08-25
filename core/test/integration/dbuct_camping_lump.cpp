// Verifies the lump-deposit invariant: after a child's frame exhausts its budget
// of K episodes, the parent's bank.visits increases by exactly K and bank.value
// increases by exactly the sum of the K rewards passed to terminate().  The
// caller observes both the reward it supplied and the resulting bank.value delta;
// the assertion is that they match exactly.  Both bank.get_visits() and
// bank.get_value() are caller-owned public surfaces.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctCampingLumpTest : public ::testing::Test
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

    // Runs one episode and returns the reward value passed to terminate().
    // Reward is pre-initialised from the starting position so that camping
    // episodes at an in-bounds node carry a non-zero base reward.
    double run_episode(manifest_t&                m,
                       const std::vector<double>& track,
                       const std::vector<jump_t>& jumps,
                       std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = (position >= 0 && position < static_cast<int>(track.size()))
                          ? track[position] : 0.0;

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
                return reward;
            }
            position = next;
            reward   = track[position];
        }
    }

    struct PeriodResult { size_t delta_visits; double delta_value; double sum_rewards; };

    // Drives episodes until visits.get_visits(-1) changes and returns:
    //   delta_visits  — how much root.visits grew (equals grant_k)
    //   delta_value   — how much root.value grew
    //   sum_rewards   — sum of every reward the caller passed to terminate()
    //
    // The lump invariant asserts delta_value == sum_rewards exactly.
    PeriodResult run_grant_period(manifest_t&                m,
                                  visits_t&                  visits,
                                  value_t&                   value,
                                  const std::vector<double>& track,
                                  const std::vector<jump_t>& jumps,
                                  std::vector<int>&          path)
    {
        const size_t before_v   = visits.get_visits(-1);
        const double before_val = value.get_value(-1);
        double       sum        = 0.0;
        while (visits.get_visits(-1) == before_v)
            sum += run_episode(m, track, jumps, path);
        return {visits.get_visits(-1) - before_v,
                value.get_value(-1)   - before_val,
                sum};
    }
};

TEST_F(DbuctCampingLumpTest, LumpInvariantHoldsAcrossGrantPeriods)
{
    // track={7.0}: pos0 in-bounds (value 7.0), pos1+ OOB.
    // The lump invariant: value[root] delta == sum of all rewards
    // supplied to terminate() during that grant period.  This holds
    // regardless of nested frame depth or which episodes yield 0 reward.
    // The visit-proportional grant check: delta_visits == 1 + floor(k * V_before),
    // where V_before is root's visit count at period start (frozen for the whole
    // period, since root only banks visits when its child backsteps).
    const std::vector<double> track = {7.0};
    const std::vector<jump_t> jumps = {1};
    const double              k     = 0.5;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, k, -1);
    std::vector<int> path = {-1};

    // Loop over 10 sequential periods.
    // For each: assert the grant formula and the lump invariant.
    for (size_t period = 0; period < 10; ++period)
    {
        const size_t V_before       = visits.get_visits(-1);
        const size_t expected_grant = 1 + static_cast<size_t>(k * V_before);
        const PeriodResult r = run_grant_period(m, visits, value, track, jumps, path);
        EXPECT_EQ(r.delta_visits, expected_grant)
            << "visits delta wrong for period=" << period
            << " V_before=" << V_before << " k=" << k;
        EXPECT_DOUBLE_EQ(r.delta_value, r.sum_rewards)
            << "value lump mismatch for period=" << period
            << " (delta_value=" << r.delta_value
            << " sum_rewards=" << r.sum_rewards << ")";
    }
}
