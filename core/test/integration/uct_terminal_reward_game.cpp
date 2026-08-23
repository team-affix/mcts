// Navigate the track; reward = track[last in-bounds position before OOB].
// No intermediate coins means a single terminal reward, which is valid for node
// contraction, so this uses position_walker: positions reachable from multiple
// parents share statistics.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "optimal_scores.hpp"
#include "walkers.hpp"

class TerminalRewardGameTest : public ::testing::Test
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

    static constexpr double kTolerance = 0.001;

    double simulate_once(
        visits_t&                  visits,
        value_t&                   value,
        const std::vector<double>& track,
        const std::vector<jump_t>& jumps,
        std::mt19937&              rng,
        double                     exploration_constant)
    {
        manifest_t m(visits, visits, value, value, rng, exploration_constant, -1);

        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                break;
            }
            position = next;
            reward   = track[position];
        }

        return reward;
    }

    void verify_converges_to_optimal(
        int                        seed,
        size_t                     track_length,
        const std::vector<jump_t>& move_amounts,
        int                        training_sims)
    {
        std::mt19937                           rng(seed);
        std::uniform_real_distribution<double> urd(-10, 10);

        std::vector<double> track(track_length);
        std::generate(track.begin(), track.end(), [&] { return urd(rng); });

        std::cerr << "track:";
        for (double v : track)
            std::cerr << " " << std::fixed << std::setprecision(3) << v;
        std::cerr << "\n";

        constexpr double exploration_constant = 100.0;

        visits_t visits;
        value_t  value;

        for (int i = 0; i < training_sims; ++i)
            simulate_once(visits, value, track, move_amounts, rng, exploration_constant);

        const double exploitative_score =
            simulate_once(visits, value, track, move_amounts, rng, 0.0);
        const double optimal = optimal_last_position_score(track, move_amounts);

        EXPECT_NEAR(exploitative_score, optimal, kTolerance);
    }
};

TEST_F(TerminalRewardGameTest, Seed40Track10Moves123)
{
    verify_converges_to_optimal(40, 10, {1, 2, 3}, 10000);
}

TEST_F(TerminalRewardGameTest, Seed41Track10Moves123)
{
    verify_converges_to_optimal(41, 10, {1, 2, 3}, 10000);
}

TEST_F(TerminalRewardGameTest, Seed42Track30Moves123)
{
    verify_converges_to_optimal(42, 30, {1, 2, 3}, 200000);
}

TEST_F(TerminalRewardGameTest, Seed43Track30PrimeMoves)
{
    verify_converges_to_optimal(43, 30, {2, 3, 5, 7}, 200000);
}

TEST_F(TerminalRewardGameTest, Seed44Track10Moves25)
{
    verify_converges_to_optimal(44, 10, {2, 5}, 10000);
}

TEST_F(TerminalRewardGameTest, Seed45Track10Moves1234)
{
    verify_converges_to_optimal(45, 10, {1, 2, 3, 4}, 10000);
}

TEST_F(TerminalRewardGameTest, Seed46Track15Moves123)
{
    verify_converges_to_optimal(46, 15, {1, 2, 3}, 20000);
}

TEST_F(TerminalRewardGameTest, Seed47Track15Moves235)
{
    verify_converges_to_optimal(47, 15, {2, 3, 5}, 20000);
}

TEST_F(TerminalRewardGameTest, Seed48Track10Moves15)
{
    verify_converges_to_optimal(48, 10, {1, 5}, 10000);
}

TEST_F(TerminalRewardGameTest, Seed49Track20Moves123)
{
    verify_converges_to_optimal(49, 20, {1, 2, 3}, 50000);
}
