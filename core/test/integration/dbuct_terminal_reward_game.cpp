// The dbuct manifest on the terminal-reward game.  Uses position_walker so the
// node handle IS the position (no path accumulation); the reward is purely the
// last in-bounds position, independent of the path taken.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "optimal_scores.hpp"
#include "walkers.hpp"

class DbuctTerminalRewardGameTest : public ::testing::Test
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

    static constexpr double kTolerance = 0.001;

    void train(visits_t&                  visits,
               value_t&                   value,
               const std::vector<double>& track,
               const std::vector<jump_t>& jumps,
               std::mt19937&              rng,
               double                     exploration_constant,
               size_t                     grant_increment_interval,
               int                        training_sims)
    {
        manifest_t m(visits, visits, value, value, rng, exploration_constant,
                     grant_increment_interval, -1);

        std::vector<int> path = {-1};

        for (int i = 0; i < training_sims; ++i)
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
                    break;
                }
                position = next;
                reward   = track[position];
            }
        }
    }

    double greedy_run(visits_t&                  visits,
                      value_t&                   value,
                      const std::vector<double>& track,
                      const std::vector<jump_t>& jumps,
                      std::mt19937&              rng)
    {
        manifest_t m(visits, visits, value, value, rng, 0.0,
                     std::numeric_limits<size_t>::max(), -1);

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

    void verify_converges_to_optimal(int                        seed,
                                     size_t                     track_length,
                                     const std::vector<jump_t>& move_amounts,
                                     int                        training_sims,
                                     size_t                     gii =
                                         std::numeric_limits<size_t>::max())
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
        train(visits, value, track, move_amounts, rng, exploration_constant, gii, training_sims);

        const double exploitative_score =
            greedy_run(visits, value, track, move_amounts, rng);
        const double optimal = optimal_last_position_score(track, move_amounts);

        EXPECT_NEAR(exploitative_score, optimal, kTolerance);
    }
};

// gii = SIZE_MAX  =>  vanilla UCT.
TEST_F(DbuctTerminalRewardGameTest, VanillaGIISeed40Track10Moves123)
{
    verify_converges_to_optimal(40, 10, {1, 2, 3}, 10000);
}

TEST_F(DbuctTerminalRewardGameTest, VanillaGIISeed44Track10Moves25)
{
    verify_converges_to_optimal(44, 10, {2, 5}, 10000);
}

TEST_F(DbuctTerminalRewardGameTest, VanillaGIISeed46Track15Moves123)
{
    verify_converges_to_optimal(46, 15, {1, 2, 3}, 20000);
}

TEST_F(DbuctTerminalRewardGameTest, VanillaGIISeed49Track20Moves123)
{
    verify_converges_to_optimal(49, 20, {1, 2, 3}, 50000);
}

// Finite gii.
TEST_F(DbuctTerminalRewardGameTest, GII10Seed40Track10Moves123)
{
    verify_converges_to_optimal(40, 10, {1, 2, 3}, 10000, 10);
}

TEST_F(DbuctTerminalRewardGameTest, GII5Seed44Track10Moves25)
{
    verify_converges_to_optimal(44, 10, {2, 5}, 10000, 5);
}

TEST_F(DbuctTerminalRewardGameTest, GII3Seed46Track15Moves123)
{
    verify_converges_to_optimal(46, 15, {1, 2, 3}, 20000, 3);
}
