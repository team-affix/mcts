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

using jump_t = int;   // a jump is simply a signed distance

// Hash for std::vector<int>, enabling unordered_map keyed by path.
struct VectorIntHash
{
    size_t operator()(const std::vector<int>& v) const noexcept
    {
        size_t seed = v.size();
        for (int x : v)
            seed ^= static_cast<size_t>(x) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// Map alias used by CoinCollectingGameTest's bank_t.
template<typename K, typename V>
using path_unordered_map = std::unordered_map<K, V, VectorIntHash>;

// Walker for node-contraction tests: node handle is the current position.
struct position_walker
{
    int walk(const int& node_handle, jump_t j) const { return node_handle + j; }
};

// Walker for tree-based tests: node handle is the full path of positions from
// root, making every distinct traversal route a unique node.
struct path_walker
{
    std::vector<int> walk(const std::vector<int>& path, jump_t j) const
    {
        std::vector<int> child = path;
        child.push_back(path.back() + j);
        return child;
    }
};

// DP for the coin-collecting game: maximise the SUM of coins along the path.
double optimal_cumulative_score(const std::vector<double>& track,
                                const std::vector<jump_t>& jumps)
{
    int n = (int)track.size();
    std::vector<double> dp(n, 0.0);

    for (int pos = n - 1; pos >= 0; --pos)
    {
        double best = -std::numeric_limits<double>::infinity();
        for (jump_t j : jumps)
        {
            int next = pos + j;
            best = std::max(best, (next < n) ? track[next] + dp[next] : 0.0);
        }
        dp[pos] = best;
    }

    double best = -std::numeric_limits<double>::infinity();
    for (jump_t j : jumps)
    {
        int next = -1 + j;
        best = std::max(best, (next < n) ? track[next] + dp[next] : 0.0);
    }
    return best;
}

// DP for the terminal-reward game: maximise track[last_in_bounds_position].
double optimal_last_position_score(const std::vector<double>& track,
                                   const std::vector<jump_t>& jumps)
{
    int n = (int)track.size();
    std::vector<double> dp(n, -std::numeric_limits<double>::infinity());

    for (int pos = n - 1; pos >= 0; --pos)
    {
        for (jump_t j : jumps)
        {
            int    next      = pos + j;
            double candidate = (next >= n) ? track[pos] : dp[next];
            dp[pos]          = std::max(dp[pos], candidate);
        }
    }

    double best = -std::numeric_limits<double>::infinity();
    for (jump_t j : jumps)
    {
        int next = -1 + j;
        if (next >= 0 && next < n)
            best = std::max(best, dp[next]);
    }
    return best;
}

} // namespace

// ---------------------------------------------------------------------------
// CoinCollectingGameTest
//
// Uses path_walker so every distinct traversal route is a unique node.
// Equivalent to tree-based MCTS — no scope sharing, no value contamination.
// ---------------------------------------------------------------------------
class CoinCollectingGameTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<std::vector<int>, double, path_unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;

    static constexpr double kTolerance = 0.001;

    double simulate_once(
        bank_t&                    bank,
        const std::vector<double>& track,
        const std::vector<jump_t>& jumps,
        std::mt19937&              rng,
        double                     exploration_constant)
    {
        rollout_t   rollout(rng);
        path_walker walker;
        monte_carlo::uniform_value_delta<double> delta;

        monte_carlo::sim<
            std::vector<int>, jump_t, double,
            bank_t, bank_t, bank_t, bank_t,
            path_walker,
            std::vector<jump_t>, std::vector<jump_t>,
            rollout_t,
            monte_carlo::uniform_value_delta<double>
        > s(bank, bank, bank, bank, walker, rollout, delta,
            std::vector<int>{-1}, exploration_constant);

        int    position    = -1;
        double total_score = 0.0;

        while (true)
        {
            jump_t chosen = s.choose(jumps, jumps);
            position += chosen;
            if (position >= static_cast<int>(track.size()))
                break;
            total_score += track[position];
        }

        delta.set_value(total_score);
        s.terminate();
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

        double exploration_constant = 0.0;
        for (double c : track)
            if (c > 0) exploration_constant += c;

        bank_t bank;

        for (int i = 0; i < training_sims; ++i)
            simulate_once(bank, track, move_amounts, rng, exploration_constant);

        const double exploitative_score =
            simulate_once(bank, track, move_amounts, rng, 0.0);
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
    verify_converges_to_optimal(28, 10, {1, 2, 3}, 100000);
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
    verify_converges_to_optimal(38, 10, {2, 3}, 10000);
}

TEST_F(CoinCollectingGameTest, Seed39Track15Moves147)
{
    verify_converges_to_optimal(39, 15, {1, 4, 7}, 10000);
}

// ---------------------------------------------------------------------------
// TerminalRewardGameTest
//
// Navigate the track; reward = track[last in-bounds position before OOB].
// No intermediate coins → single terminal reward → valid for node contraction.
// Uses position_walker so positions reachable from multiple parents share stats.
// ---------------------------------------------------------------------------
class TerminalRewardGameTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;

    static constexpr double kTolerance = 0.001;

    double simulate_once(
        bank_t&                    bank,
        const std::vector<double>& track,
        const std::vector<jump_t>& jumps,
        std::mt19937&              rng,
        double                     exploration_constant)
    {
        rollout_t       rollout(rng);
        position_walker walker;
        monte_carlo::uniform_value_delta<double> delta;

        monte_carlo::sim<
            int, jump_t, double,
            bank_t, bank_t, bank_t, bank_t,
            position_walker,
            std::vector<jump_t>, std::vector<jump_t>,
            rollout_t,
            monte_carlo::uniform_value_delta<double>
        > s(bank, bank, bank, bank, walker, rollout, delta, -1, exploration_constant);

        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = s.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= static_cast<int>(track.size()))
            {
                delta.set_value(reward);
                s.terminate();
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

        double exploration_constant = 0.0;
        for (double c : track)
            if (c > 0) exploration_constant += c;

        bank_t bank;

        for (int i = 0; i < training_sims; ++i)
            simulate_once(bank, track, move_amounts, rng, exploration_constant);

        const double exploitative_score =
            simulate_once(bank, track, move_amounts, rng, 0.0);
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
