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
    using visits_t  = monte_carlo::visits_table<std::vector<int>, path_unordered_map>;
    using value_t   = monte_carlo::value_table<std::vector<int>, double, path_unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;

    static constexpr double kTolerance = 0.001;

    double simulate_once(
        visits_t&                  visits,
        value_t&                   value,
        const std::vector<double>& track,
        const std::vector<jump_t>& jumps,
        std::mt19937&              rng,
        double                     exploration_constant)
    {
        rollout_t   rollout(rng);
        path_walker walker;
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(exploration_constant);

        monte_carlo::sim<
            std::vector<int>, jump_t, double,
            visits_t, value_t, visits_t, value_t,
            path_walker,
            std::vector<jump_t>, std::vector<jump_t>,
            rollout_t,
            monte_carlo::uniform_value_delta<double>,
            monte_carlo::uniform_exploration_constant<double>
        > s(visits, value, visits, value, walker, rollout, delta, ec,
            std::vector<int>{-1});

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
    using visits_t  = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t   = monte_carlo::value_table<int, double, std::unordered_map>;
    using rollout_t = monte_carlo::random_rollout<
                         jump_t, std::mt19937,
                         std::vector<jump_t>, std::vector<jump_t>>;

    static constexpr double kTolerance = 0.001;

    double simulate_once(
        visits_t&                  visits,
        value_t&                   value,
        const std::vector<double>& track,
        const std::vector<jump_t>& jumps,
        std::mt19937&              rng,
        double                     exploration_constant)
    {
        rollout_t       rollout(rng);
        position_walker walker;
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(exploration_constant);

        monte_carlo::sim<
            int, jump_t, double,
            visits_t, value_t, visits_t, value_t,
            position_walker,
            std::vector<jump_t>, std::vector<jump_t>,
            rollout_t,
            monte_carlo::uniform_value_delta<double>,
            monte_carlo::uniform_exploration_constant<double>
        > s(visits, value, visits, value, walker, rollout, delta, ec, -1);

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
    using visits_t      = monte_carlo::visits_table<std::vector<int>, path_unordered_map>;
    using value_t       = monte_carlo::value_table<std::vector<int>, double, path_unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<std::vector<int>, path_unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             std::vector<int>, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             path_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

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
        rollout_t        rollout(rng);
        path_walker      walker;
        dispatches_t     dispatches;
        batch_t          batch(grant_increment_interval);
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(exploration_constant);
        std::vector<int> root = {-1};

        dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
                  walker, rollout, delta, ec, root);

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

            delta.set_value(ep_score);
            d.terminate();
            path.resize(d.depth());
        }
    }

    double greedy_run(visits_t&                  visits,
                      value_t&                   value,
                      const std::vector<double>& track,
                      const std::vector<jump_t>& jumps,
                      std::mt19937&              rng)
    {
        rollout_t        rollout(rng);
        path_walker      walker;
        dispatches_t     dispatches;
        batch_t          batch(std::numeric_limits<size_t>::max());
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(0.0);
        std::vector<int> root = {-1};

        dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
                  walker, rollout, delta, ec, root);

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

        delta.set_value(ep_score);
        d.terminate();
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

// ---------------------------------------------------------------------------
// DbuctTerminalRewardGameTest
//
// Uses position_walker so node handle == position (no path accumulation).
// Reward is purely the last in-bounds position, independent of path taken.
// ---------------------------------------------------------------------------
class DbuctTerminalRewardGameTest : public ::testing::Test
{
protected:
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

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
        rollout_t       rollout(rng);
        position_walker walker;
        dispatches_t    dispatches;
        batch_t         batch(grant_increment_interval);
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(exploration_constant);

        dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
                  walker, rollout, delta, ec, -1);

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
                    delta.set_value(reward);
                    d.terminate();
                    path.resize(d.depth());
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
        rollout_t       rollout(rng);
        position_walker walker;
        dispatches_t    dispatches;
        batch_t         batch(std::numeric_limits<size_t>::max());
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(0.0);

        dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
                  walker, rollout, delta, ec, -1);

        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = d.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= static_cast<int>(track.size()))
            {
                delta.set_value(reward);
                d.terminate();
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
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

    // One sim episode using the terminal-reward convention (reward = last
    // in-bounds position's track value).
    void sim_episode(visits_t&                  visits,
                     value_t&                   value,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::mt19937&              rng,
                     double                     c)
    {
        rollout_t       rollout(rng);
        position_walker walker;
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(c);

        monte_carlo::sim<
            int, jump_t, double,
            visits_t, value_t, visits_t, value_t,
            position_walker,
            std::vector<jump_t>, std::vector<jump_t>,
            rollout_t,
            monte_carlo::uniform_value_delta<double>,
            monte_carlo::uniform_exploration_constant<double>
        > s(visits, value, visits, value, walker, rollout, delta, ec, -1);

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
    void dbuct_episodes(visits_t&                  visits,
                        value_t&                   value,
                        const std::vector<double>& track,
                        const std::vector<jump_t>& jumps,
                        std::mt19937&              rng,
                        double                     c,
                        int                        n)
    {
        rollout_t       rollout(rng);
        position_walker walker;
        dispatches_t    dispatches;
        batch_t         batch(std::numeric_limits<size_t>::max());
        monte_carlo::uniform_value_delta<double>        delta;
        monte_carlo::uniform_exploration_constant<double> ec(c);

        dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
                  walker, rollout, delta, ec, -1);

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
                    delta.set_value(reward);
                    d.terminate();
                    path.resize(d.depth());
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
    visits_t     sim_visits,   dbuct_visits;
    value_t      sim_value,    dbuct_value;

    for (int i = 0; i < N; ++i)
        sim_episode(sim_visits, sim_value, track, jumps, rng1, c);
    dbuct_episodes(dbuct_visits, dbuct_value, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(sim_visits.get_visits(pos), dbuct_visits.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(sim_value.get_value(pos), dbuct_value.get_value(pos))
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
    visits_t     sim_visits,   dbuct_visits;
    value_t      sim_value,    dbuct_value;

    for (int i = 0; i < N; ++i)
        sim_episode(sim_visits, sim_value, track, jumps, rng1, c);
    dbuct_episodes(dbuct_visits, dbuct_value, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(sim_visits.get_visits(pos), dbuct_visits.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(sim_value.get_value(pos), dbuct_value.get_value(pos))
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
    visits_t     sim_visits,   dbuct_visits;
    value_t      sim_value,    dbuct_value;

    for (int i = 0; i < N; ++i)
        sim_episode(sim_visits, sim_value, track, jumps, rng1, c);
    dbuct_episodes(dbuct_visits, dbuct_value, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(sim_visits.get_visits(pos), dbuct_visits.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(sim_value.get_value(pos), dbuct_value.get_value(pos))
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
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;
};

TEST_F(DbuctInRolloutTest, FlagTransitionsEpisodes1And2)
{
    // track={5.0}: positions root(-1), pos0(0), OOB at 1.
    // Every episode uses exactly two choose() calls before terminate().
    const std::vector<double> track = {5.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;
    rollout_t                 rollout(rng);
    position_walker           walker;
    dispatches_t              dispatches;
    batch_t                   batch(std::numeric_limits<size_t>::max());
    monte_carlo::uniform_value_delta<double>        delta;
    monte_carlo::uniform_exploration_constant<double> ec(1.0);

    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, delta, ec, -1);

    // Episode 1: root has 0 visits → in_rollout flips on the very first choose().
    EXPECT_FALSE(d.in_rollout());

    d.choose(jumps, jumps);        // at root (0 visits): immediate rollout, no frame pushed
    EXPECT_TRUE(d.in_rollout());   // flipped at expansion (root itself is expansion node)

    d.choose(jumps, jumps);        // still in rollout (pos0 → jump → OOB next)
    EXPECT_TRUE(d.in_rollout());   // flag persists until terminate()

    delta.set_value(5.0);
    d.terminate();                 // resets flag
    EXPECT_FALSE(d.in_rollout());

    // Episode 2: root (1 visit) → UCB → pos0 pushed; pos0 (0 visits) → expansion.
    EXPECT_FALSE(d.in_rollout());

    d.choose(jumps, jumps);        // UCB at root (1 visit): tree phase, pos0 frame pushed
    EXPECT_FALSE(d.in_rollout()); // still tree phase — flag not set during UCB selection

    d.choose(jumps, jumps);        // at pos0 (0 visits): expansion, rollout chosen
    EXPECT_TRUE(d.in_rollout());  // flipped exactly here

    delta.set_value(5.0);
    d.terminate();
    EXPECT_FALSE(d.in_rollout());
}

// ---------------------------------------------------------------------------
// DbuctDepthTest
//
// Verifies depth() after terminate() reflects budget-driven backtracking.
// Root only = 1; a child camping one level deep = 2.
// Callers sync their path via path.resize(depth()).
// ---------------------------------------------------------------------------
class DbuctDepthTest : public ::testing::Test
{
protected:
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

    monte_carlo::uniform_value_delta<double> delta;

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
                delta.set_value(reward);
                d.terminate();
                path.resize(d.depth());
                return;
            }
            position = next;
            reward   = track[position];
        }
    }
};

TEST_F(DbuctDepthTest, ReturnsCorrectCampingFrameIndex)
{
    // Single-step game: root(-1) → pos0 → OOB.
    // GII=2: dispatches D=0,1 give grant=1 (budget=1, always fully consumed → depth=1).
    //        dispatch  D=2     gives grant=2 (budget=2 at pos0):
    //          first  episode under that budget: pos0 not exhausted → camping, depth=2.
    //          second episode under that budget: pos0 exhausted     → backstep to root, depth=1.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;
    rollout_t                 rollout(rng);
    position_walker           walker;
    dispatches_t              dispatches;
    batch_t                   batch(2);   // GII = 2

    monte_carlo::uniform_exploration_constant<double> ec(0.0);
    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, delta, ec, -1);

    std::vector<int> path = {-1};

    // Expansion-first: choose() always dispatches a child before rolling out,
    // so every episode increments the dispatch counter.
    // D=0 → grant=1, D=1 → grant=1, D=2 → grant=2 (camping begins).

    // ep1: D(-1)=0 before dispatch, grant=1. Expand pos0, rollout from pos0.
    //      pos0 budget=1 exhausted immediately → depth=1, path={-1}.
    run_episode(d, track, jumps, path);
    EXPECT_EQ(d.depth(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep2: D(-1)=1 before dispatch, grant=1. Same pattern → depth=1, path={-1}.
    run_episode(d, track, jumps, path);
    EXPECT_EQ(d.depth(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep3: D(-1)=2 before dispatch, grant=2. pos0 gets budget=2; after 1 sim
    //      visit_lump=1<2 → camping at pos0, depth=2, path={-1, 0}.
    run_episode(d, track, jumps, path);
    EXPECT_EQ(d.depth(), 2u);
    EXPECT_EQ(path.back(), 0);

    // ep4: continuing from pos0 (path={-1,0}). pos0's second sim exhausts budget=2
    //      → backstep to root, depth=1, path={-1}.
    run_episode(d, track, jumps, path);
    EXPECT_EQ(d.depth(), 1u);
    EXPECT_EQ(path.back(), -1);
}

TEST_F(DbuctDepthTest, ManualBackstepToRootOverridesCamping)
{
    // Same GII=2 setup as above.  Episode 4 would normally camp at pos0
    // (depth=2).  Caller invokes backstep() after terminate() to climb to root,
    // and the pos0 lump is rolled into root's lump as usual.
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;
    rollout_t                 rollout(rng);
    position_walker           walker;
    dispatches_t              dispatches;
    batch_t                   batch(2);   // GII = 2

    monte_carlo::uniform_exploration_constant<double> ec(0.0);
    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, delta, ec, -1);

    std::vector<int> path = {-1};

    // Advance through eps 1-2 (D=0,1 → grant=1 periods; camping begins at D=2).
    run_episode(d, track, jumps, path);
    run_episode(d, track, jumps, path);
    ASSERT_EQ(path.back(), -1);

    const size_t root_visits_before = visits.get_visits(-1);

    // ep4: grant=2, pos0 would camp at depth=2 — caller backsteps to root.
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
                delta.set_value(reward);
                d.terminate();
                while (d.depth() > 1)
                    d.backstep();
                path.resize(d.depth());
                break;
            }
            position = next;
            reward   = track[position];
        }
    }

    EXPECT_EQ(d.depth(), 1u);
    EXPECT_EQ(path.back(), -1);

    // Verify that pos0's partial lump (1 visit) was rolled into root even
    // though pos0's budget was not naturally exhausted.
    EXPECT_EQ(visits.get_visits(-1) - root_visits_before, 1u);
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
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

    monte_carlo::uniform_value_delta<double> delta;

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
                delta.set_value(reward);
                d.terminate();
                path.resize(d.depth());
                return;
            }
            position = next;
            reward   = track[position];
        }
    }

    // Runs episodes until visits.get_visits(-1) increases, then returns the delta.
    // The delta equals the grant_k assigned to the child that just backstep-ed.
    size_t run_grant_period(dbuct_t&                   d,
                            visits_t&                  visits,
                            const std::vector<double>& track,
                            const std::vector<jump_t>& jumps,
                            std::vector<int>&          path)
    {
        size_t before = visits.get_visits(-1);
        while (visits.get_visits(-1) == before)
            run_episode(d, track, jumps, path);
        return visits.get_visits(-1) - before;
    }
};

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithRootDispatchesGII3)
{
    // Single-path game: root → pos0 → OOB.
    // Each grant period root dispatches pos0 exactly once (D increments by 1).
    // grant_k = 1 + D_before / GII, verified via the delta in visits.get_visits(-1).
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 3;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;
    dispatches_t              dispatches;
    batch_t                   batch(GII);
    rollout_t                 rollout(rng);
    position_walker           walker;

    monte_carlo::uniform_exploration_constant<double> ec(0.0);
    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, delta, ec, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit (rollout phase; no UCB dispatch happens here).
    run_grant_period(d, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    // From here, every period involves exactly one UCB dispatch from root.
    // Formula: grant_k = 1 + D_before / GII (integer division).
    for (size_t period = 0; period < 10; ++period)
    {
        const size_t D_before = dispatches.get_dispatches(-1);
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(d, visits, track, jumps, path);
        EXPECT_EQ(dispatches.get_dispatches(-1), D_before + 1)
            << "dispatch count did not increment at period=" << period;
        EXPECT_EQ(visits.get_visits(-1) - V_before, 1 + D_before / GII)
            << "grant mismatch at period=" << period
            << " D_before=" << D_before << " GII=" << GII;
    }
}

TEST_F(DbuctGrantFormulaTest, GrantGrowsWithRootDispatchesGII5)
{
    const std::vector<double> track = {1.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 5;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;
    dispatches_t              dispatches;
    batch_t                   batch(GII);
    rollout_t                 rollout(rng);
    position_walker           walker;

    monte_carlo::uniform_exploration_constant<double> ec(0.0);
    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, delta, ec, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit (rollout phase; no UCB dispatch happens here).
    run_grant_period(d, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    // GII=5: grant stays 1 for D=0..4, then rises by 1 every 5 dispatches.
    for (size_t period = 0; period < 12; ++period)
    {
        const size_t D_before = dispatches.get_dispatches(-1);
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(d, visits, track, jumps, path);
        EXPECT_EQ(dispatches.get_dispatches(-1), D_before + 1)
            << "dispatch count did not increment at period=" << period;
        EXPECT_EQ(visits.get_visits(-1) - V_before, 1 + D_before / GII)
            << "grant mismatch at period=" << period
            << " D_before=" << D_before << " GII=" << GII;
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
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

    monte_carlo::uniform_value_delta<double> delta;

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
                delta.set_value(reward);
                d.terminate();
                path.resize(d.depth());
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
    PeriodResult run_grant_period(dbuct_t&                   d,
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
            sum += run_episode(d, track, jumps, path);
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
    // The dispatch-based grant check: delta_visits == 1 + D_before / GII.
    const std::vector<double> track = {7.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 2;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;
    dispatches_t              dispatches;
    batch_t                   batch(GII);
    rollout_t                 rollout(rng);
    position_walker           walker;

    monte_carlo::uniform_exploration_constant<double> ec(0.0);
    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, delta, ec, -1);
    std::vector<int> path = {-1};

    // Loop over 10 sequential periods.
    // For each: assert dispatch-based grant and lump invariant.
    for (size_t period = 0; period < 10; ++period)
    {
        const size_t D_before        = dispatches.get_dispatches(-1);
        const size_t expected_grant  = 1 + D_before / GII;
        const PeriodResult r = run_grant_period(d, visits, value, track, jumps, path);
        EXPECT_EQ(r.delta_visits, expected_grant)
            << "visits delta wrong for period=" << period
            << " D_before=" << D_before << " GII=" << GII;
        EXPECT_DOUBLE_EQ(r.delta_value, r.sum_rewards)
            << "value lump mismatch for period=" << period
            << " (delta_value=" << r.delta_value
            << " sum_rewards=" << r.sum_rewards << ")";
    }
}

// ---------------------------------------------------------------------------
// DbuctBudgetInvariantTest
//
// DISABLED: verifying budget(parent(n)) >= budget(n) during backstep()
// requires reading each frame's grant budget, which is intentionally not
// exposed on the public API. Without a budget accessor (or a test-only hook),
// the invariant cannot be asserted from caller code.
// ---------------------------------------------------------------------------
#if 0
class DbuctBudgetInvariantTest : public ::testing::Test
{
protected:
    using visits_t      = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t       = monte_carlo::value_table<int, double, std::unordered_map>;
    using dispatches_t  = monte_carlo::dispatches_table<int, std::unordered_map>;
    using batch_t       = monte_carlo::linear_batch_increment;
    using rollout_t     = monte_carlo::random_rollout<
                             jump_t, std::mt19937,
                             std::vector<jump_t>, std::vector<jump_t>>;
    using dbuct_t       = monte_carlo::dbuct<
                             int, jump_t, double,
                             visits_t, value_t, visits_t, value_t,
                             dispatches_t, dispatches_t,
                             batch_t,
                             position_walker,
                             std::vector<jump_t>, std::vector<jump_t>,
                             rollout_t,
                             monte_carlo::uniform_value_delta<double>,
                             monte_carlo::uniform_exploration_constant<double>>;

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
                d.terminate(reward);
                path.resize(d.depth());
                return;
            }
            position = next;
            reward   = track[position];
        }
    }

    void backstep_to_root_asserting_invariant(dbuct_t& d, std::vector<int>& path)
    {
        while (d.depth() > 1)
        {
            const size_t child_budget = d.budget();
            d.backstep();
            EXPECT_GE(d.budget(), child_budget)
                << "budget(parent) >= budget(child) violated at depth=" << d.depth();
            path.resize(d.depth());
        }
    }
};

TEST_F(DbuctBudgetInvariantTest, ParentBudgetGteChildBudgetAcrossLongRun)
{
    const std::vector<double> track = {3.0, -1.0, 4.0, -2.0, 5.0, 1.0, 2.0, -3.0, 7.0, 0.5};
    const std::vector<jump_t> jumps = {1, 2, 3};
    const size_t              GII   = 3;
    std::mt19937              rng(77);
    visits_t                  visits;
    value_t                   value;
    dispatches_t              dispatches;
    batch_t                   batch(GII);
    rollout_t                 rollout(rng);
    position_walker           walker;

    dbuct_t d(visits, value, visits, value, dispatches, dispatches, batch,
              walker, rollout, -1, 1.5);

    std::vector<int> path = {-1};

    for (int i = 0; i < 10000; ++i)
        run_episode(d, track, jumps, path);

    for (int i = 0; i < 1000; ++i)
    {
        run_episode(d, track, jumps, path);
        backstep_to_root_asserting_invariant(d, path);
        EXPECT_EQ(d.depth(), 1u);
        EXPECT_EQ(path, std::vector<int>({-1}));
    }
}
#endif
