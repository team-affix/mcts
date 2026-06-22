#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mcts.hpp"

class CoinCollectingGameTest : public ::testing::Test
{
protected:
    struct jump
    {
        int m_amount = 0;
        bool operator<(const jump& a_other) const { return m_amount < a_other.m_amount; }
    };

    static constexpr double kTolerance = 0.001;

    double simulate_coin_collecting_game(
        monte_carlo::tree_node<jump>& a_root,
        const std::vector<double>&    a_track,
        const std::vector<int>&       a_jump_lengths,
        std::mt19937&                 a_rnd_dev)
    {
        double l_exploration_constant = 0;
        for (double l_coin : a_track)
            if (l_coin > 0) l_exploration_constant += l_coin;

        monte_carlo::simulation<jump, std::mt19937> l_sim(a_root, l_exploration_constant, a_rnd_dev);

        int    l_position     = -1;
        double l_total_score  = 0;
        size_t l_action_count = 0;

        std::vector<jump> l_actions;
        std::transform(a_jump_lengths.begin(), a_jump_lengths.end(), std::back_inserter(l_actions),
            [](int a_jl) { return jump{ a_jl }; });

        while (true)
        {
            jump l_chosen_action = l_sim.choose(l_actions);
            ++l_action_count;
            l_position += l_chosen_action.m_amount;
            if (l_position >= static_cast<int>(a_track.size())) break;
            l_total_score += a_track[l_position];
        }

        l_sim.terminate(l_total_score);
        EXPECT_EQ(l_sim.length(), l_action_count);

        return l_total_score;
    }

    double run_coin_collecting_simulations(
        int                         a_seed,
        size_t                      a_track_length,
        const std::vector<int>&     a_moves,
        int                         a_simulations)
    {
        std::mt19937                           l_mt19937(a_seed);
        std::uniform_real_distribution<double> l_urd(-10, 10);

        std::vector<double> l_track(a_track_length);
        std::generate(l_track.begin(), l_track.end(), [&l_mt19937, &l_urd] { return l_urd(l_mt19937); });

        monte_carlo::tree_node<jump> l_root{};
        double                       l_score = 0;

        for (int i = 0; i < a_simulations; ++i)
            l_score = simulate_coin_collecting_game(l_root, l_track, a_moves, l_mt19937);

        return l_score;
    }
};

TEST_F(CoinCollectingGameTest, Seed27Track10Moves123)
{
    const double l_score = run_coin_collecting_simulations(27, 10, { 1, 2, 3 }, 10000);
    EXPECT_NEAR(l_score, 26.91873636, kTolerance);
}

TEST_F(CoinCollectingGameTest, Seed28Track10Moves123)
{
    const double l_score = run_coin_collecting_simulations(28, 10, { 1, 2, 3 }, 100000);
    EXPECT_NEAR(l_score, 10.145513480, kTolerance);
}

TEST_F(CoinCollectingGameTest, Seed29Track30Moves123)
{
    const double l_score = run_coin_collecting_simulations(29, 30, { 1, 2, 3 }, 100000);
    EXPECT_NEAR(l_score, 52.135578, kTolerance);
}

TEST_F(CoinCollectingGameTest, Seed30Track30PrimeMoves)
{
    const double l_score = run_coin_collecting_simulations(30, 30, { 2, 3, 5, 7 }, 100000);
    EXPECT_NEAR(l_score, 45.596276158, kTolerance);
}
