// The full uct manifest must converge to the DP optimum on the coin-collecting
// game.  Uses path_walker so every distinct traversal route is a unique node,
// which is equivalent to tree-based MCTS: no scope sharing, no value
// contamination.

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

class CoinCollectingGameTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<std::vector<int>, path_unordered_map>;
    using value_t    = monte_carlo::value_table<std::vector<int>, double, path_unordered_map>;
    using manifest_t = monte_carlo::uct_value_manifest<
                          std::vector<int>, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          path_walker,
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
        manifest_t m(visits, visits, value, value, rng, exploration_constant,
                     std::vector<int>{-1});

        int    position    = -1;
        double total_score = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            position += chosen;
            if (position >= static_cast<int>(track.size()))
                break;
            total_score += track[position];
        }

        m.delta.set_value(total_score);
        m.terminator.terminate();
        return total_score;
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
        const double optimal = optimal_cumulative_score(track, move_amounts);

        EXPECT_NEAR(exploitative_score, optimal, kTolerance);
    }
};

TEST_F(CoinCollectingGameTest, Seed27Track10Moves123)
{
    verify_converges_to_optimal(27, 10, {1, 2, 3}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed28Track10Moves123)
{
    verify_converges_to_optimal(28, 10, {1, 2, 3}, 2000000);
}

TEST_F(CoinCollectingGameTest, Seed31Track10Moves25)
{
    verify_converges_to_optimal(31, 10, {2, 5}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed32Track10Moves1234)
{
    verify_converges_to_optimal(32, 10, {1, 2, 3, 4}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed34Track15Moves235)
{
    verify_converges_to_optimal(34, 15, {2, 3, 5}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed35Track10Moves15)
{
    verify_converges_to_optimal(35, 10, {1, 5}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed36Track20Moves123)
{
    verify_converges_to_optimal(36, 20, {1, 2, 3}, 50000);
}

TEST_F(CoinCollectingGameTest, Seed37Track20Moves357)
{
    verify_converges_to_optimal(37, 20, {3, 5, 7}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed38Track10Moves23)
{
    verify_converges_to_optimal(38, 10, {2, 3}, 50000);
}

TEST_F(CoinCollectingGameTest, Seed39Track15Moves147)
{
    verify_converges_to_optimal(39, 15, {1, 4, 7}, 10000);
}
