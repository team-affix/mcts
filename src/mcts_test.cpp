#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "mcts.hpp"

namespace
{

struct jump
{
    int m_amount = 0;
    bool operator<(const jump& a_other) const { return m_amount < a_other.m_amount; }
};

struct position_walker
{
    int walk(const int& node_handle, jump j) const { return node_handle + j.m_amount; }
};

// Arrival reward: the coin collected when landing at walker(node, choice).
// Returns 0 when the destination is out of bounds (terminal step).
struct track_arrival_reward
{
    track_arrival_reward(const std::vector<double>& track) : track_(track) {}

    double get_arrival_reward(const int& node, jump j) const
    {
        int next = node + j.m_amount;
        return (next < static_cast<int>(track_.size())) ? track_[next] : 0.0;
    }

private:
    const std::vector<double>& track_;
};

// Exact optimal score for the coin-collecting game via DP.
// dp[pos] = max future score achievable from position `pos` (not counting the
// coin already collected upon arriving at pos).
// Terminal when the next position >= track.size().
double optimal_score(const std::vector<double>& track, const std::vector<jump>& jumps)
{
    int n = (int)track.size();
    std::vector<double> dp(n, 0.0);

    for (int pos = n - 1; pos >= 0; --pos)
    {
        double best = -std::numeric_limits<double>::infinity();
        for (const jump& j : jumps)
        {
            int next = pos + j.m_amount;
            best = std::max(best, (next < n) ? track[next] + dp[next] : 0.0);
        }
        dp[pos] = best;
    }

    double best = -std::numeric_limits<double>::infinity();
    for (const jump& j : jumps)
    {
        int next = -1 + j.m_amount;
        best = std::max(best, (next < n) ? track[next] + dp[next] : 0.0);
    }
    return best;
}

} // namespace

class CoinCollectingGameTest : public ::testing::Test
{
protected:
    using bank_t     = monte_carlo::map_table<int, double, std::unordered_map>;
    using edge_t     = monte_carlo::int_edge_unordered_table<int>;
    using arrival_t  = track_arrival_reward;
    using rollout_t  = monte_carlo::random_rollout<
                           jump, std::mt19937, std::vector<jump>, std::vector<jump>>;

    static constexpr double kTolerance = 0.001;

    double simulate_once(
        bank_t&                    bank,
        edge_t&                    edges,
        const std::vector<double>& track,
        const std::vector<jump>&   jumps,
        std::mt19937&              rng,
        double                     exploration_constant)
    {
        arrival_t       arrival(track);
        rollout_t       rollout(rng);
        position_walker walker;

        monte_carlo::sim<
            int, jump, double,
            bank_t, bank_t, bank_t, bank_t,
            position_walker, arrival_t,
            edge_t, edge_t,
            std::vector<jump>, std::vector<jump>,
            rollout_t
        > s(bank, bank, bank, bank, walker, arrival, edges, edges, rollout, -1,
            exploration_constant);

        auto& jumps_ref = const_cast<std::vector<jump>&>(jumps);

        int    position    = -1;
        double total_score = 0;

        while (true)
        {
            jump chosen = s.choose(jumps_ref, jumps_ref);
            position += chosen.m_amount;
            if (position >= static_cast<int>(track.size()))
            {
                s.step_terminal();
                break;
            }
            total_score += track[position];
            s.step_reward(track[position]);
        }

        s.terminate();
        return total_score;
    }

    // Trains for `training_sims` then runs one greedy (c=0) exploitation pass.
    // Asserts that the greedy score matches the DP-computed optimal.
    void verify_converges_to_optimal(
        int                     seed,
        size_t                  track_length,
        const std::vector<int>& move_amounts,
        int                     training_sims)
    {
        std::mt19937                           rng(seed);
        std::uniform_real_distribution<double> urd(-10, 10);

        std::vector<double> track(track_length);
        std::generate(track.begin(), track.end(), [&] { return urd(rng); });

        std::cerr << "track:";
        for (double v : track)
            std::cerr << " " << std::fixed << std::setprecision(3) << v;
        std::cerr << "\n";

        std::vector<jump> jumps;
        for (int a : move_amounts) jumps.push_back(jump{a});

        double exploration_constant = 0;
        for (double c : track)
            if (c > 0) exploration_constant += c;

        bank_t bank;
        edge_t edges;

        for (int i = 0; i < training_sims; ++i)
            simulate_once(bank, edges, track, jumps, rng, exploration_constant);

        // Pure exploitation: c=0 picks the highest-value child at every step.
        const double exploitative_score = simulate_once(bank, edges, track, jumps, rng, 0.0);
        const double optimal_store = optimal_score(track, jumps);

        EXPECT_NEAR(exploitative_score, optimal_store, kTolerance);
    }
};

TEST_F(CoinCollectingGameTest, Seed27Track10Moves123)
{
    verify_converges_to_optimal(27, 10, {1, 2, 3}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed28Track10Moves123)
{
    verify_converges_to_optimal(28, 10, {1, 2, 3}, 100000);
}

TEST_F(CoinCollectingGameTest, Seed29Track30Moves123)
{
    verify_converges_to_optimal(29, 30, {1, 2, 3}, 800000);
}

TEST_F(CoinCollectingGameTest, Seed30Track30PrimeMoves)
{
    verify_converges_to_optimal(30, 30, {2, 3, 5, 7}, 800000);
}

TEST_F(CoinCollectingGameTest, Seed31Track10Moves25)
{
    verify_converges_to_optimal(31, 10, {2, 5}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed32Track10Moves1234)
{
    verify_converges_to_optimal(32, 10, {1, 2, 3, 4}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed33Track15Moves123)
{
    verify_converges_to_optimal(33, 15, {1, 2, 3}, 10000);
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
    verify_converges_to_optimal(36, 20, {1, 2, 3}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed37Track20Moves357)
{
    verify_converges_to_optimal(37, 20, {3, 5, 7}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed38Track10Moves23)
{
    verify_converges_to_optimal(38, 10, {2, 3}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed39Track15Moves147)
{
    verify_converges_to_optimal(39, 15, {1, 4, 7}, 10000);
}
