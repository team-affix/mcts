// The full dbuct manifest must converge to the DP optimum on the coin-collecting
// game, both with an infinite grant increment interval (equivalent to vanilla
// UCT) and with finite ones.  Uses path_walker so every distinct traversal route
// is a unique node; the reward passed to terminate() is the full root-to-terminal
// coin sum so UCB statistics stay globally comparable at every depth.

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

class DbuctCoinCollectingGameTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<std::vector<int>, path_unordered_map>;
    using value_t    = monte_carlo::value_table<std::vector<int>, double, path_unordered_map>;
    using manifest_t = monte_carlo::dbuct_value_manifest<
                          std::vector<int>, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          path_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937, path_unordered_map>;

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
        std::vector<int> root = {-1};

        manifest_t m(visits, visits, value, value, rng, exploration_constant,
                     grant_increment_interval, root);

        std::vector<int> path = root;

        for (int i = 0; i < training_sims; ++i)
        {
            double base_score = 0.0;
            for (int pos : path)
                if (pos >= 0 && pos < static_cast<int>(track.size()))
                    base_score += track[pos];

            int    position = path.back();
            double ep_score = base_score;

            while (true)
            {
                jump_t chosen = m.chooser.choose(jumps, jumps);
                position += chosen;
                if (!m.in_rollout.is_in_rollout())
                    path.push_back(position);
                if (position >= static_cast<int>(track.size()))
                    break;
                ep_score += track[position];
            }

            m.delta.set_value(ep_score);
            m.terminator.terminate();
            path.resize(m.frame_stack.size());
        }
    }

    double greedy_run(visits_t&                  visits,
                      value_t&                   value,
                      const std::vector<double>& track,
                      const std::vector<jump_t>& jumps,
                      std::mt19937&              rng)
    {
        std::vector<int> root = {-1};

        manifest_t m(visits, visits, value, value, rng, 0.0,
                     std::numeric_limits<size_t>::max(), root);

        int    position = -1;
        double ep_score = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            position += chosen;
            if (position >= static_cast<int>(track.size()))
                break;
            ep_score += track[position];
        }

        m.delta.set_value(ep_score);
        m.terminator.terminate();
        return ep_score;
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
        const double optimal = optimal_cumulative_score(track, move_amounts);

        EXPECT_NEAR(exploitative_score, optimal, kTolerance);
    }
};

// gii = SIZE_MAX  =>  vanilla UCT; same parameters as CoinCollectingGameTest.
TEST_F(DbuctCoinCollectingGameTest, VanillaGIISeed27Track10Moves123)
{
    verify_converges_to_optimal(27, 10, {1, 2, 3}, 10000);
}

TEST_F(DbuctCoinCollectingGameTest, VanillaGIISeed31Track10Moves25)
{
    verify_converges_to_optimal(31, 10, {2, 5}, 10000);
}

TEST_F(DbuctCoinCollectingGameTest, VanillaGIISeed34Track15Moves235)
{
    verify_converges_to_optimal(34, 15, {2, 3, 5}, 10000);
}

TEST_F(DbuctCoinCollectingGameTest, VanillaGIISeed36Track20Moves123)
{
    verify_converges_to_optimal(36, 20, {1, 2, 3}, 50000);
}

// Finite gii — algorithm still converges, budget efficiency differs.
TEST_F(DbuctCoinCollectingGameTest, GII10Seed27Track10Moves123)
{
    verify_converges_to_optimal(27, 10, {1, 2, 3}, 10000, 10);
}

TEST_F(DbuctCoinCollectingGameTest, GII5Seed31Track10Moves25)
{
    verify_converges_to_optimal(31, 10, {2, 5}, 10000, 5);
}

TEST_F(DbuctCoinCollectingGameTest, GII3Seed34Track15Moves235)
{
    verify_converges_to_optimal(34, 15, {2, 3, 5}, 20000, 3);
}
