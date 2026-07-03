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

// ---------------------------------------------------------------------------
// DbuctCoinCollectingGameTest
//
// Uses path_walker so every distinct traversal route is a unique node.
// The reward passed to terminate() is the full root-to-terminal coin sum,
// so that UCB statistics at every depth remain globally comparable.
// ---------------------------------------------------------------------------
class DbuctCoinCollectingGameTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<std::vector<int>, double, path_unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         std::vector<int>, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         path_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;

    static constexpr double kTolerance = 0.001;

    void train(bank_t&                    bank,
               const std::vector<double>& track,
               const std::vector<jump_t>& jumps,
               std::mt19937&              rng,
               double                     exploration_constant,
               size_t                     grant_increment_interval,
               int                        training_sims)
    {
        rollout_t        rollout(rng);
        path_walker      walker;
        std::vector<int> root = {-1};

        dbuct_t d(bank, bank, bank, bank, walker, rollout, root,
                  grant_increment_interval, exploration_constant);

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
                jump_t chosen = d.choose(jumps, jumps);
                position += chosen;
                if (!d.in_rollout())
                    path.push_back(position);
                if (position >= static_cast<int>(track.size()))
                    break;
                ep_score += track[position];
            }

            size_t steps = d.terminate(ep_score);
            for (size_t s = 0; s < steps; ++s)
                path.pop_back();
        }
    }

    double greedy_run(bank_t&                    bank,
                      const std::vector<double>& track,
                      const std::vector<jump_t>& jumps,
                      std::mt19937&              rng)
    {
        rollout_t        rollout(rng);
        path_walker      walker;
        std::vector<int> root = {-1};

        dbuct_t d(bank, bank, bank, bank, walker, rollout, root,
                  std::numeric_limits<size_t>::max(), 0.0);

        int    position = -1;
        double ep_score = 0.0;

        while (true)
        {
            jump_t chosen = d.choose(jumps, jumps);
            position += chosen;
            if (position >= static_cast<int>(track.size()))
                break;
            ep_score += track[position];
        }

        d.terminate(ep_score);
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

        double exploration_constant = 0.0;
        for (double c : track)
            if (c > 0) exploration_constant += c;

        bank_t bank;
        train(bank, track, move_amounts, rng, exploration_constant, gii, training_sims);

        const double exploitative_score =
            greedy_run(bank, track, move_amounts, rng);
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

// ---------------------------------------------------------------------------
// DbuctTerminalRewardGameTest
//
// Uses position_walker so node handle == position (no path accumulation).
// Reward is purely the last in-bounds position, independent of path taken.
// ---------------------------------------------------------------------------
class DbuctTerminalRewardGameTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         int, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         position_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;

    static constexpr double kTolerance = 0.001;

    void train(bank_t&                    bank,
               const std::vector<double>& track,
               const std::vector<jump_t>& jumps,
               std::mt19937&              rng,
               double                     exploration_constant,
               size_t                     grant_increment_interval,
               int                        training_sims)
    {
        rollout_t       rollout(rng);
        position_walker walker;

        dbuct_t d(bank, bank, bank, bank, walker, rollout, -1,
                  grant_increment_interval, exploration_constant);

        std::vector<int> path = {-1};

        for (int i = 0; i < training_sims; ++i)
        {
            int    position = path.back();
            double reward   = 0.0;

            while (true)
            {
                jump_t chosen = d.choose(jumps, jumps);
                int    next   = position + chosen;
                if (!d.in_rollout())
                    path.push_back(next);
                if (next >= static_cast<int>(track.size()))
                {
                    size_t steps = d.terminate(reward);
                    for (size_t s = 0; s < steps; ++s)
                        path.pop_back();
                    break;
                }
                position = next;
                reward   = track[position];
            }
        }
    }

    double greedy_run(bank_t&                    bank,
                      const std::vector<double>& track,
                      const std::vector<jump_t>& jumps,
                      std::mt19937&              rng)
    {
        rollout_t       rollout(rng);
        position_walker walker;

        dbuct_t d(bank, bank, bank, bank, walker, rollout, -1,
                  std::numeric_limits<size_t>::max(), 0.0);

        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = d.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= static_cast<int>(track.size()))
            {
                d.terminate(reward);
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

        double exploration_constant = 0.0;
        for (double c : track)
            if (c > 0) exploration_constant += c;

        bank_t bank;
        train(bank, track, move_amounts, rng, exploration_constant, gii, training_sims);

        const double exploitative_score =
            greedy_run(bank, track, move_amounts, rng);
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

// ---------------------------------------------------------------------------
// DbuctStatEquivalenceTest
//
// With GII=SIZE_MAX dbuct must produce bit-identical bank stats to vanilla sim
// after N episodes driven from the same RNG state.  All assertions operate
// exclusively on caller-owned bank queries: get_visits() and get_value().
// ---------------------------------------------------------------------------
class DbuctStatEquivalenceTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         int, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         position_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;

    // One sim episode using the terminal-reward convention (reward = last
    // in-bounds position's track value).
    void sim_episode(bank_t&                    bank,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::mt19937&              rng,
                     double                     c)
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
        > s(bank, bank, bank, bank, walker, rollout, delta, -1, c);

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
    }

    // N dbuct episodes with GII=SIZE_MAX (≡ vanilla UCT).
    void dbuct_episodes(bank_t&                    bank,
                        const std::vector<double>& track,
                        const std::vector<jump_t>& jumps,
                        std::mt19937&              rng,
                        double                     c,
                        int                        n)
    {
        rollout_t       rollout(rng);
        position_walker walker;

        dbuct_t d(bank, bank, bank, bank, walker, rollout, -1,
                  std::numeric_limits<size_t>::max(), c);

        std::vector<int> path = {-1};

        for (int i = 0; i < n; ++i)
        {
            int    position = path.back();
            double reward   = 0.0;

            while (true)
            {
                jump_t chosen = d.choose(jumps, jumps);
                int    next   = position + chosen;
                if (!d.in_rollout())
                    path.push_back(next);
                if (next >= static_cast<int>(track.size()))
                {
                    size_t steps = d.terminate(reward);
                    for (size_t s = 0; s < steps; ++s)
                        path.pop_back();
                    break;
                }
                position = next;
                reward   = track[position];
            }
        }
    }
};

TEST_F(DbuctStatEquivalenceTest, MatchesSimSeed100Track5Moves12)
{
    const std::vector<double> track = {3.0, 1.0, 4.0, 1.0, 5.0};
    const std::vector<jump_t> jumps = {1, 2};
    const double              c     = 5.0;
    const int                 N     = 200;

    std::mt19937 rng1(100), rng2(100);
    bank_t       bank_sim, bank_dbuct;

    for (int i = 0; i < N; ++i)
        sim_episode(bank_sim, track, jumps, rng1, c);
    dbuct_episodes(bank_dbuct, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(bank_sim.get_visits(pos), bank_dbuct.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(bank_sim.get_value(pos), bank_dbuct.get_value(pos))
            << "value mismatch at pos=" << pos;
    }
}

TEST_F(DbuctStatEquivalenceTest, MatchesSimSeed200Track4Moves13)
{
    const std::vector<double> track = {2.0, 7.0, 1.0, 8.0};
    const std::vector<jump_t> jumps = {1, 3};
    const double              c     = 8.0;
    const int                 N     = 300;

    std::mt19937 rng1(200), rng2(200);
    bank_t       bank_sim, bank_dbuct;

    for (int i = 0; i < N; ++i)
        sim_episode(bank_sim, track, jumps, rng1, c);
    dbuct_episodes(bank_dbuct, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(bank_sim.get_visits(pos), bank_dbuct.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(bank_sim.get_value(pos), bank_dbuct.get_value(pos))
            << "value mismatch at pos=" << pos;
    }
}

TEST_F(DbuctStatEquivalenceTest, MatchesSimSeed300Track6Moves123)
{
    const std::vector<double> track = {9.0, 2.0, 6.0, 5.0, 3.0, 5.0};
    const std::vector<jump_t> jumps = {1, 2, 3};
    const double              c     = 9.0;
    const int                 N     = 500;

    std::mt19937 rng1(300), rng2(300);
    bank_t       bank_sim, bank_dbuct;

    for (int i = 0; i < N; ++i)
        sim_episode(bank_sim, track, jumps, rng1, c);
    dbuct_episodes(bank_dbuct, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(bank_sim.get_visits(pos), bank_dbuct.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(bank_sim.get_value(pos), bank_dbuct.get_value(pos))
            << "value mismatch at pos=" << pos;
    }
}

// ---------------------------------------------------------------------------
// DbuctInRolloutTest
//
// Verifies the in_rollout() state machine via its public accessor:
//   - false before any choose() in an episode
//   - flips to true exactly when the expansion node is encountered
//   - resets to false after terminate()
// ---------------------------------------------------------------------------
class DbuctInRolloutTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         int, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         position_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;
};

TEST_F(DbuctInRolloutTest, FlagTransitionsEpisodes1And2)
{
    // track={5.0}: positions root(-1), pos0(0), OOB at 1.
    // Every episode uses exactly two choose() calls before terminate().
    const std::vector<double> track = {5.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    bank_t                    bank;
    rollout_t                 rollout(rng);
    position_walker           walker;

    dbuct_t d(bank, bank, bank, bank, walker, rollout, -1,
              std::numeric_limits<size_t>::max(), 1.0);

    // Episode 1: root has 0 visits → in_rollout flips on the very first choose().
    EXPECT_FALSE(d.in_rollout());

    d.choose(jumps, jumps);        // at root (0 visits): immediate rollout, no frame pushed
    EXPECT_TRUE(d.in_rollout());   // flipped at expansion (root itself is expansion node)

    d.choose(jumps, jumps);        // still in rollout (pos0 → jump → OOB next)
    EXPECT_TRUE(d.in_rollout());   // flag persists until terminate()

    d.terminate(5.0);              // resets flag
    EXPECT_FALSE(d.in_rollout());

    // Episode 2: root (1 visit) → UCB → pos0 pushed; pos0 (0 visits) → expansion.
    EXPECT_FALSE(d.in_rollout());

    d.choose(jumps, jumps);        // UCB at root (1 visit): tree phase, pos0 frame pushed
    EXPECT_FALSE(d.in_rollout()); // still tree phase — flag not set during UCB selection

    d.choose(jumps, jumps);        // at pos0 (0 visits): expansion, rollout chosen
    EXPECT_TRUE(d.in_rollout());  // flipped exactly here

    d.terminate(5.0);
    EXPECT_FALSE(d.in_rollout());
}

// ---------------------------------------------------------------------------
// DbuctBackstepCountTest
//
// Verifies the return value of terminate() equals the exact number of frames
// popped from the internal stack, observable as the depth of the tree policy
// traversal that just completed.
// ---------------------------------------------------------------------------
class DbuctBackstepCountTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         int, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         position_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;

    size_t run_episode(dbuct_t&                   d,
                       const std::vector<double>& track,
                       const std::vector<jump_t>& jumps,
                       std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = d.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!d.in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                size_t steps = d.terminate(reward);
                for (size_t s = 0; s < steps; ++s)
                    path.pop_back();
                return steps;
            }
            position = next;
            reward   = track[position];
        }
    }
};

TEST_F(DbuctBackstepCountTest, GrowingDepthGIIMax)
{
    // track={1,1,1}: positions 0,1,2 in-bounds; 3=OOB.
    // With GII=SIZE_MAX every child frame gets budget=1 and pops after one episode.
    // The tree grows one level per episode: ep1→0 steps, ep2→1, ep3→2, ep4→3.
    const std::vector<double> track = {1.0, 1.0, 1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    bank_t                    bank;
    rollout_t                 rollout(rng);
    position_walker           walker;

    dbuct_t d(bank, bank, bank, bank, walker, rollout, -1,
              std::numeric_limits<size_t>::max(), 0.0);

    std::vector<int> path = {-1};

    EXPECT_EQ(run_episode(d, track, jumps, path), 0u); // root rollout, no frames popped
    EXPECT_EQ(run_episode(d, track, jumps, path), 1u); // UCB depth 1 → pos0 frame popped
    EXPECT_EQ(run_episode(d, track, jumps, path), 2u); // UCB depth 2 → pos1 + pos0 popped
    EXPECT_EQ(run_episode(d, track, jumps, path), 3u); // UCB depth 3 → pos2+pos1+pos0 popped
}

// ---------------------------------------------------------------------------
// DbuctGrantFormulaTest
//
// Verifies grant_k = 1 + N / GII for several N values by observing the delta
// in bank.get_visits(root) after each camping grant period completes.
// Root's visit count only increases when a child frame backsteps to root, and
// increases by exactly grant_k (the child's budget), making the jump in
// bank.get_visits(-1) the sole public observable needed.
// ---------------------------------------------------------------------------
class DbuctGrantFormulaTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         int, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         position_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;

    void run_episode(dbuct_t&                   d,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = d.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!d.in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                size_t steps = d.terminate(reward);
                for (size_t s = 0; s < steps; ++s)
                    path.pop_back();
                return;
            }
            position = next;
            reward   = track[position];
        }
    }

    // Runs episodes until bank.get_visits(-1) increases, then returns the delta.
    // The delta equals the grant_k assigned to the child that just backstep-ed.
    size_t run_grant_period(dbuct_t&                   d,
                            bank_t&                    bank,
                            const std::vector<double>& track,
                            const std::vector<jump_t>& jumps,
                            std::vector<int>&          path)
    {
        size_t before = bank.get_visits(-1);
        while (bank.get_visits(-1) == before)
            run_episode(d, track, jumps, path);
        return bank.get_visits(-1) - before;
    }
};

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithRootVisitsGII3)
{
    // Single-path game: root → pos0 → OOB.
    // root's bank.visits only increments in bulk when pos0's frame backsteps.
    // The bulk increment equals pos0's budget = 1 + N/GII (N = visits at dispatch).
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 3;
    std::mt19937              rng(0);
    bank_t                    bank;
    rollout_t                 rollout(rng);
    position_walker           walker;

    dbuct_t d(bank, bank, bank, bank, walker, rollout, -1, GII, 0.0);
    std::vector<int> path = {-1};

    // Each row: {expected root.visits before the period, expected grant_k}.
    // Formula: grant_k = 1 + N/GII (integer division).
    //   N=0→1, N=1→1, N=2→1, N=3→2, N=5→2, N=7→3, N=10→4
    const std::vector<std::pair<size_t, size_t>> cases = {
        {0, 1}, {1, 1}, {2, 1}, {3, 2}, {5, 2}, {7, 3}, {10, 4}
    };

    for (const auto& kv : cases)
    {
        const size_t expected_N     = kv.first;
        const size_t expected_grant = kv.second;
        ASSERT_EQ(bank.get_visits(-1), expected_N)
            << "precondition: root should have " << expected_N << " visits";
        const size_t actual_grant = run_grant_period(d, bank, track, jumps, path);
        EXPECT_EQ(actual_grant, expected_grant)
            << "grant mismatch for N=" << expected_N << " GII=" << GII;
    }
}

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithRootVisitsGII5)
{
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 5;
    std::mt19937              rng(0);
    bank_t                    bank;
    rollout_t                 rollout(rng);
    position_walker           walker;

    dbuct_t d(bank, bank, bank, bank, walker, rollout, -1, GII, 0.0);
    std::vector<int> path = {-1};

    // GII=5: grant stays 1 for N=0..4, then rises by 1 every 5 root visits.
    //   N=0→1, N=1→1, N=2→1, N=3→1, N=4→1, N=5→2, N=7→2, N=9→2, N=11→3
    const std::vector<std::pair<size_t, size_t>> cases = {
        {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1},
        {5, 2}, {7, 2}, {9, 2}, {11, 3}
    };

    for (const auto& kv : cases)
    {
        const size_t expected_N     = kv.first;
        const size_t expected_grant = kv.second;
        ASSERT_EQ(bank.get_visits(-1), expected_N)
            << "precondition: root should have " << expected_N << " visits";
        const size_t actual_grant = run_grant_period(d, bank, track, jumps, path);
        EXPECT_EQ(actual_grant, expected_grant)
            << "grant mismatch for N=" << expected_N << " GII=" << GII;
    }
}

// ---------------------------------------------------------------------------
// DbuctCampingLumpTest
//
// Verifies the lump-deposit invariant: after a child's frame exhausts its
// budget of K episodes, the parent's bank.visits increases by exactly K and
// bank.value increases by exactly the sum of the K rewards passed to
// terminate().  The caller observes both the reward it supplied and the
// resulting bank.value delta; the assertion is that they match exactly.
// Both bank.get_visits() and bank.get_value() are caller-owned public surfaces.
// ---------------------------------------------------------------------------
class DbuctCampingLumpTest : public ::testing::Test
{
protected:
    using bank_t    = monte_carlo::map_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t   = monte_carlo::dbuct<
                         int, jump_t, double,
                         bank_t, bank_t, bank_t, bank_t,
                         position_walker,
                         std::vector<jump_t>, std::vector<jump_t>,
                         rollout_t>;

    // Runs one episode and returns the reward value passed to terminate().
    // Reward is pre-initialised from the starting position so that camping
    // episodes at an in-bounds node carry a non-zero base reward.
    double run_episode(dbuct_t&                   d,
                       const std::vector<double>& track,
                       const std::vector<jump_t>& jumps,
                       std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = (position >= 0 && position < static_cast<int>(track.size()))
                          ? track[position] : 0.0;

        while (true)
        {
            jump_t chosen = d.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!d.in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                size_t steps = d.terminate(reward);
                for (size_t s = 0; s < steps; ++s)
                    path.pop_back();
                return reward;
            }
            position = next;
            reward   = track[position];
        }
    }

    struct PeriodResult { size_t delta_visits; double delta_value; double sum_rewards; };

    // Drives episodes until bank.get_visits(-1) changes and returns:
    //   delta_visits  — how much root.visits grew (equals grant_k)
    //   delta_value   — how much root.value grew
    //   sum_rewards   — sum of every reward the caller passed to terminate()
    //
    // The lump invariant asserts delta_value == sum_rewards exactly.
    PeriodResult run_grant_period(dbuct_t&                   d,
                                  bank_t&                    bank,
                                  const std::vector<double>& track,
                                  const std::vector<jump_t>& jumps,
                                  std::vector<int>&          path)
    {
        const size_t before_v   = bank.get_visits(-1);
        const double before_val = bank.get_value(-1);
        double       sum        = 0.0;
        while (bank.get_visits(-1) == before_v)
            sum += run_episode(d, track, jumps, path);
        return {bank.get_visits(-1) - before_v,
                bank.get_value(-1)  - before_val,
                sum};
    }
};

TEST_F(DbuctCampingLumpTest, LumpInvariantHoldsAcrossGrantPeriods)
{
    // track={7.0}: pos0 in-bounds (value 7.0), pos1+ OOB.
    // The lump invariant: bank.value[root] delta == sum of all rewards
    // supplied to terminate() during that grant period.  This holds
    // regardless of nested frame depth or which episodes yield 0 reward.
    const std::vector<double> track = {7.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 2;
    std::mt19937              rng(0);
    bank_t                    bank;
    rollout_t                 rollout(rng);
    position_walker           walker;

    dbuct_t d(bank, bank, bank, bank, walker, rollout, -1, GII, 0.0);
    std::vector<int> path = {-1};

    // GII=2: grant = 1 + N/2.
    //   N=0→1, N=1→1, N=2→2, N=4→3, N=7→4 (7/2=3, so 1+3=4)
    const std::vector<std::pair<size_t, size_t>> cases = {
        {0, 1}, {1, 1}, {2, 2}, {4, 3}, {7, 4}
    };

    for (const auto& kv : cases)
    {
        const size_t expected_N     = kv.first;
        const size_t expected_grant = kv.second;
        ASSERT_EQ(bank.get_visits(-1), expected_N)
            << "precondition: root should have " << expected_N << " visits";
        const PeriodResult r = run_grant_period(d, bank, track, jumps, path);
        EXPECT_EQ(r.delta_visits, expected_grant)
            << "visits delta wrong for N=" << expected_N << " GII=" << GII;
        EXPECT_DOUBLE_EQ(r.delta_value, r.sum_rewards)
            << "value lump mismatch for N=" << expected_N
            << " GII=" << GII << " (delta_value=" << r.delta_value
            << " sum_rewards=" << r.sum_rewards << ")";
    }
}
