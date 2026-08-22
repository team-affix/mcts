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
#include <gmock/gmock.h>

#include "mcts.hpp"

using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

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
    using visits_t   = monte_carlo::visits_table<std::vector<int>, path_unordered_map>;
    using value_t    = monte_carlo::value_table<std::vector<int>, double, path_unordered_map>;
    using manifest_t = monte_carlo::sim_value_manifest<
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
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::sim_value_manifest<
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
                if (!m.in_rollout.get_in_rollout())
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

// ---------------------------------------------------------------------------
// DbuctTerminalRewardGameTest
//
// Uses position_walker so node handle == position (no path accumulation).
// Reward is purely the last in-bounds position, independent of path taken.
// ---------------------------------------------------------------------------
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
                if (!m.in_rollout.get_in_rollout())
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
    using visits_t        = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t         = monte_carlo::value_table<int, double, std::unordered_map>;
    using sim_value_manifest_t = monte_carlo::sim_value_manifest<
                               int, jump_t, double,
                               visits_t, visits_t, value_t, value_t,
                               position_walker,
                               std::vector<jump_t>, std::vector<jump_t>,
                               std::mt19937>;
    using dbuct_value_manifest_t = monte_carlo::dbuct_value_manifest<
                               int, jump_t, double,
                               visits_t, visits_t, value_t, value_t,
                               position_walker,
                               std::vector<jump_t>, std::vector<jump_t>,
                               std::mt19937, std::unordered_map>;

    // One sim episode using the terminal-reward convention (reward = last
    // in-bounds position's track value).
    void sim_episode(visits_t&                  visits,
                     value_t&                   value,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::mt19937&              rng,
                     double                     c)
    {
        sim_value_manifest_t m(visits, visits, value, value, rng, c, -1);

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
        dbuct_value_manifest_t m(visits, visits, value, value, rng, c,
                           std::numeric_limits<size_t>::max(), -1);

        std::vector<int> path = {-1};

        for (int i = 0; i < n; ++i)
        {
            int    position = path.back();
            double reward   = 0.0;

            while (true)
            {
                jump_t chosen = m.chooser.choose(jumps, jumps);
                int    next   = position + chosen;
                if (!m.in_rollout.get_in_rollout())
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
// Verifies the in_rollout flag via its public collaborator:
//   - false before any choose() in an episode
//   - flips to true exactly when the expansion node is encountered
//   - resets to false after terminate()
// ---------------------------------------------------------------------------
class DbuctInRolloutTest : public ::testing::Test
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

    manifest_t m(visits, visits, value, value, rng, 1.0,
                 std::numeric_limits<size_t>::max(), -1);

    // Episode 1: root has 0 visits → in_rollout flips on the very first choose().
    EXPECT_FALSE(m.in_rollout.get_in_rollout());

    m.chooser.choose(jumps, jumps);        // at root (0 visits): immediate rollout, no frame pushed
    EXPECT_TRUE(m.in_rollout.get_in_rollout());   // flipped at expansion (root itself is expansion node)

    m.chooser.choose(jumps, jumps);        // still in rollout (pos0 → jump → OOB next)
    EXPECT_TRUE(m.in_rollout.get_in_rollout());   // flag persists until terminate()

    m.delta.set_value(5.0);
    m.terminator.terminate();   // resets flag
    EXPECT_FALSE(m.in_rollout.get_in_rollout());

    // Episode 2: root (1 visit) → UCB → pos0 pushed; pos0 (0 visits) → expansion.
    EXPECT_FALSE(m.in_rollout.get_in_rollout());

    m.chooser.choose(jumps, jumps);        // UCB at root (1 visit): tree phase, pos0 frame pushed
    EXPECT_FALSE(m.in_rollout.get_in_rollout()); // still tree phase — flag not set during UCB selection

    m.chooser.choose(jumps, jumps);        // at pos0 (0 visits): expansion, rollout chosen
    EXPECT_TRUE(m.in_rollout.get_in_rollout());  // flipped exactly here

    m.delta.set_value(5.0);
    m.terminator.terminate();
    EXPECT_FALSE(m.in_rollout.get_in_rollout());
}

// ---------------------------------------------------------------------------
// DbuctDepthTest
//
// Verifies the frame stack size after terminate() reflects budget-driven
// backtracking.  Root only = 1; a child camping one level deep = 2.
// Callers sync their path via path.resize(frame_stack.size()).
// ---------------------------------------------------------------------------
class DbuctDepthTest : public ::testing::Test
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

    void run_episode(manifest_t&                m,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.get_in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                path.resize(m.frame_stack.size());
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

    manifest_t m(visits, visits, value, value, rng, 0.0, 2, -1);   // GII = 2

    std::vector<int> path = {-1};

    // Expansion-first: choose() always dispatches a child before rolling out,
    // so every episode increments the dispatch counter.
    // D=0 → grant=1, D=1 → grant=1, D=2 → grant=2 (camping begins).

    // ep1: D(-1)=0 before dispatch, grant=1. Expand pos0, rollout from pos0.
    //      pos0 budget=1 exhausted immediately → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep2: D(-1)=1 before dispatch, grant=1. Same pattern → depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_EQ(path.back(), -1);

    // ep3: D(-1)=2 before dispatch, grant=2. pos0 gets budget=2; after 1 sim
    //      visit_lump=1<2 → camping at pos0, depth=2, path={-1, 0}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 2u);
    EXPECT_EQ(path.back(), 0);

    // ep4: continuing from pos0 (path={-1,0}). pos0's second sim exhausts budget=2
    //      → backstep to root, depth=1, path={-1}.
    run_episode(m, track, jumps, path);
    EXPECT_EQ(m.frame_stack.size(), 1u);
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

    manifest_t m(visits, visits, value, value, rng, 0.0, 2, -1);   // GII = 2

    std::vector<int> path = {-1};

    // Advance through eps 1-2 (D=0,1 → grant=1 periods; camping begins at D=2).
    run_episode(m, track, jumps, path);
    run_episode(m, track, jumps, path);
    ASSERT_EQ(path.back(), -1);

    const size_t root_visits_before = visits.get_visits(-1);

    // ep4: grant=2, pos0 would camp at depth=2 — caller backsteps to root.
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.get_in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                while (m.frame_stack.size() > 1)
                    m.value_stack_controller.backstep();
                path.resize(m.frame_stack.size());
                break;
            }
            position = next;
            reward   = track[position];
        }
    }

    EXPECT_EQ(m.frame_stack.size(), 1u);
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
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::dbuct_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937, std::unordered_map>;

    void run_episode(manifest_t&                m,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::vector<int>&          path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.get_in_rollout())
                path.push_back(next);
            if (next >= static_cast<int>(track.size()))
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                path.resize(m.frame_stack.size());
                return;
            }
            position = next;
            reward   = track[position];
        }
    }

    // Runs episodes until visits.get_visits(-1) increases, then returns the delta.
    // The delta equals the grant_k assigned to the child that just backstep-ed.
    size_t run_grant_period(manifest_t&                m,
                            visits_t&                  visits,
                            const std::vector<double>& track,
                            const std::vector<jump_t>& jumps,
                            std::vector<int>&          path)
    {
        size_t before = visits.get_visits(-1);
        while (visits.get_visits(-1) == before)
            run_episode(m, track, jumps, path);
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

    manifest_t m(visits, visits, value, value, rng, 0.0, GII, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit (rollout phase; no UCB dispatch happens here).
    run_grant_period(m, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    // From here, every period involves exactly one UCB dispatch from root.
    // Formula: grant_k = 1 + D_before / GII (integer division).
    for (size_t period = 0; period < 10; ++period)
    {
        const size_t D_before = m.dispatches.get_dispatches(-1);
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(m.dispatches.get_dispatches(-1), D_before + 1)
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

    manifest_t m(visits, visits, value, value, rng, 0.0, GII, -1);
    std::vector<int> path = {-1};

    // Seed root's initial visit (rollout phase; no UCB dispatch happens here).
    run_grant_period(m, visits, track, jumps, path);
    ASSERT_EQ(visits.get_visits(-1), 1u) << "expected root to have 1 visit after seed";

    // GII=5: grant stays 1 for D=0..4, then rises by 1 every 5 dispatches.
    for (size_t period = 0; period < 12; ++period)
    {
        const size_t D_before = m.dispatches.get_dispatches(-1);
        const size_t V_before = visits.get_visits(-1);
        run_grant_period(m, visits, track, jumps, path);
        EXPECT_EQ(m.dispatches.get_dispatches(-1), D_before + 1)
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
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::dbuct_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937, std::unordered_map>;

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
            if (!m.in_rollout.get_in_rollout())
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
    // The dispatch-based grant check: delta_visits == 1 + D_before / GII.
    const std::vector<double> track = {7.0};
    const std::vector<jump_t> jumps = {1};
    const size_t              GII   = 2;
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, GII, -1);
    std::vector<int> path = {-1};

    // Loop over 10 sequential periods.
    // For each: assert dispatch-based grant and lump invariant.
    for (size_t period = 0; period < 10; ++period)
    {
        const size_t D_before        = m.dispatches.get_dispatches(-1);
        const size_t expected_grant  = 1 + D_before / GII;
        const PeriodResult r = run_grant_period(m, visits, value, track, jumps, path);
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
// DbuctVisitAdderTest
// ---------------------------------------------------------------------------
struct MockGetTopFrame
{
    MOCK_METHOD((monte_carlo::dbuct_frame<int>&), top, (), ());
};

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockSetVisits
{
    MOCK_METHOD(void, set_visits, (const int&, size_t), ());
};

class DbuctVisitAdderTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_frame<int>           frame_{7, 10};
    NiceMock<MockGetTopFrame>               get_top_frame;
    NiceMock<MockGetVisits>                 get_visits;
    StrictMock<MockSetVisits>               set_visits;
    monte_carlo::dbuct_visit_adder<int,
                                   MockGetTopFrame,
                                   MockGetVisits,
                                   MockSetVisits> sut{get_top_frame, get_visits, set_visits};
};

TEST_F(DbuctVisitAdderTest, AddVisitsIncrementsBankAndFrameLump)
{
    frame_.visit_lump = 2;
    ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(5));
    EXPECT_CALL(set_visits, set_visits(7, 8));

    sut.add_visits(3);

    EXPECT_EQ(frame_.visit_lump, 5u);
}

// ---------------------------------------------------------------------------
// DbuctValueAdderTest
// ---------------------------------------------------------------------------
struct MockGetTopValueFrame
{
    MOCK_METHOD((monte_carlo::dbuct_value_frame<int, double>&), top, (), ());
};

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockSetValue
{
    MOCK_METHOD(void, set_value, (const int&, double), ());
};

struct MockAddValue
{
    MOCK_METHOD(void, add_value, (double), ());
};

class DbuctValueAdderTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_value_frame<int, double> value_frame_{3};
    NiceMock<MockGetTopValueFrame>              get_top_value_frame;
    NiceMock<MockGetValue>                      get_value;
    StrictMock<MockSetValue>                    set_value;
    monte_carlo::dbuct_value_adder<int, double,
                                   MockGetTopValueFrame,
                                   MockGetValue,
                                   MockSetValue> sut{get_top_value_frame, get_value, set_value};
};

TEST_F(DbuctValueAdderTest, AddValueIncrementsBankAndFrameLump)
{
    value_frame_.value_lump = 1.0;
    ON_CALL(get_top_value_frame, top()).WillByDefault(ReturnRef(value_frame_));
    ON_CALL(get_value, get_value(3)).WillByDefault(Return(4.0));
    EXPECT_CALL(set_value, set_value(3, 6.5));

    sut.add_value(2.5);

    EXPECT_DOUBLE_EQ(value_frame_.value_lump, 3.5);
}

// ---------------------------------------------------------------------------
// DbuctVisitCreditorTest
// ---------------------------------------------------------------------------
struct MockAddVisits
{
    MOCK_METHOD(void, add_visits, (size_t), ());
};

class DbuctVisitCreditorTest : public ::testing::Test
{
protected:
    StrictMock<MockAddVisits>                    visit_adder;
    monte_carlo::dbuct_visit_creditor<MockAddVisits> sut{visit_adder};
};

TEST_F(DbuctVisitCreditorTest, CreditDelegatesSingleVisit)
{
    EXPECT_CALL(visit_adder, add_visits(1)).Times(1);
    sut.credit();
}

// ---------------------------------------------------------------------------
// DbuctValueCreditorTest
// ---------------------------------------------------------------------------
struct MockCreditVisit
{
    MOCK_METHOD(void, credit, (), ());
};

struct MockGetValueDelta
{
    MOCK_METHOD(double, get_value_delta, (const int&), (const));
};

class DbuctValueCreditorTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_value_frame<int, double> value_frame_{9};
    StrictMock<MockCreditVisit>                 visit_creditor;
    NiceMock<MockGetTopValueFrame>              get_top_value_frame;
    StrictMock<MockAddValue>                    value_adder;
    NiceMock<MockGetValueDelta>                 value_delta;
    monte_carlo::dbuct_value_creditor<MockCreditVisit,
                                      MockGetTopValueFrame,
                                      MockAddValue,
                                      MockGetValueDelta> sut{
        visit_creditor, get_top_value_frame, value_adder, value_delta};
};

TEST_F(DbuctValueCreditorTest, CreditVisitsThenAddsDelta)
{
    ON_CALL(get_top_value_frame, top()).WillByDefault(ReturnRef(value_frame_));
    ON_CALL(value_delta, get_value_delta(9)).WillByDefault(Return(2.5));

    InSequence seq;
    EXPECT_CALL(visit_creditor, credit());
    EXPECT_CALL(value_adder, add_value(2.5));

    sut.credit();
}

// ---------------------------------------------------------------------------
// DbuctTerminatorTest
// ---------------------------------------------------------------------------
struct MockBackstep
{
    MOCK_METHOD(void, backstep, (), ());
};

struct MockValueCreditor
{
    MOCK_METHOD(void, credit, (), ());
};

struct MockSetInRollout
{
    MOCK_METHOD(void, set_in_rollout, (bool), ());
};

class DbuctTerminatorTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_frame<int> frame_{0, 0};
    NiceMock<MockGetTopFrame>     get_top_frame;
    StrictMock<MockBackstep>      backstep;
    StrictMock<MockValueCreditor> value_creditor;
    StrictMock<MockSetInRollout>  set_in_rollout;
};

TEST_F(DbuctTerminatorTest, TerminateCreditsThenBackstepsWhileBudgetExhausted)
{
    frame_.budget     = 2;
    frame_.visit_lump = 3;
    ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));

    EXPECT_CALL(value_creditor, credit()).Times(1);
    EXPECT_CALL(backstep, backstep())
        .Times(2)
        .WillOnce([&] { frame_.visit_lump = 2; })
        .WillOnce([&] { frame_.visit_lump = 0; });
    EXPECT_CALL(set_in_rollout, set_in_rollout(false)).Times(1);

    monte_carlo::dbuct_terminator<MockBackstep,
                                  MockGetTopFrame,
                                  MockValueCreditor,
                                  MockSetInRollout> sut{
        backstep, get_top_frame, value_creditor, set_in_rollout};

    sut.terminate();
}

TEST_F(DbuctTerminatorTest, TerminateSkipsBackstepWhenCamping)
{
    frame_.budget     = 3;
    frame_.visit_lump = 1;
    ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));

    EXPECT_CALL(value_creditor, credit()).Times(1);
    EXPECT_CALL(backstep, backstep()).Times(0);
    EXPECT_CALL(set_in_rollout, set_in_rollout(false)).Times(1);

    monte_carlo::dbuct_terminator<MockBackstep,
                                  MockGetTopFrame,
                                  MockValueCreditor,
                                  MockSetInRollout> sut{
        backstep, get_top_frame, value_creditor, set_in_rollout};

    sut.terminate();
}

// ---------------------------------------------------------------------------
// DbuctChooserTest
// ---------------------------------------------------------------------------
struct MockGetDispatches
{
    MOCK_METHOD(size_t, get_dispatches, (const int&), (const));
};

struct MockSetDispatches
{
    MOCK_METHOD(void, set_dispatches, (const int&, size_t), ());
};

struct MockComputeBatchSize
{
    MOCK_METHOD(size_t, compute_batch_size, (size_t), (const));
};

struct MockForestep
{
    MOCK_METHOD(void, forestep, (const monte_carlo::dbuct_frame<int>&), ());
};

struct MockPolicyChoose
{
    MOCK_METHOD(int, policy_choose, (const int&, const std::vector<jump_t>&, const std::vector<jump_t>&), ());
};

struct MockRolloutChoose
{
    MOCK_METHOD(jump_t, rollout_choose, (const std::vector<jump_t>&, const std::vector<jump_t>&), ());
};

struct MockGetInRollout
{
    MOCK_METHOD(bool, get_in_rollout, (), (const));
};

struct MockSetInRolloutChooser
{
    MOCK_METHOD(void, set_in_rollout, (bool), ());
};

class DbuctChooserTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_frame<int> frame_{-1, 100};
    NiceMock<MockGetVisits>       get_visits;
    NiceMock<MockGetDispatches>   get_dispatches;
    StrictMock<MockSetDispatches> set_dispatches;
    NiceMock<MockComputeBatchSize> compute_batch_size;
    StrictMock<MockForestep>      forestep;
    NiceMock<MockGetTopFrame>     get_top_frame;
    position_walker               walker;
    StrictMock<MockPolicyChoose>  policy;
    StrictMock<MockRolloutChoose> rollout;
    NiceMock<MockGetInRollout>    get_in_rollout;
    StrictMock<MockSetInRolloutChooser> set_in_rollout;
    std::vector<jump_t>           jumps_{1};

    monte_carlo::dbuct_chooser<int, jump_t,
                                 MockGetVisits,
                                 MockGetDispatches,
                                 MockSetDispatches,
                                 MockComputeBatchSize,
                                 MockForestep,
                                 MockGetTopFrame,
                                 position_walker,
                                 std::vector<jump_t>,
                                 std::vector<jump_t>,
                                 MockPolicyChoose,
                                 MockRolloutChoose,
                                 MockGetInRollout,
                                 MockSetInRolloutChooser> sut{
        get_visits,
        get_dispatches,
        set_dispatches,
        compute_batch_size,
        forestep,
        get_top_frame,
        walker,
        policy,
        rollout,
        get_in_rollout,
        set_in_rollout};

    void expect_tree_frame()
    {
        ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));
    }
};

TEST_F(DbuctChooserTest, RolloutPhaseDelegatesToRolloutChoose)
{
    ON_CALL(get_in_rollout, get_in_rollout()).WillByDefault(Return(true));
    EXPECT_CALL(rollout, rollout_choose(jumps_, jumps_)).WillOnce(Return(jump_t{1}));
    EXPECT_CALL(policy, policy_choose(_, _, _)).Times(0);
    EXPECT_CALL(forestep, forestep(_)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(DbuctChooserTest, TreePhaseDispatchesAndForesteps)
{
    expect_tree_frame();
    ON_CALL(get_in_rollout, get_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_dispatches, get_dispatches(-1)).WillByDefault(Return(0));
    ON_CALL(compute_batch_size, compute_batch_size(0)).WillByDefault(Return(5));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(5));

    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(set_dispatches, set_dispatches(-1, 1));
    EXPECT_CALL(forestep, forestep(_));
    EXPECT_CALL(rollout, rollout_choose(_, _)).Times(0);
    EXPECT_CALL(set_in_rollout, set_in_rollout(true)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(DbuctChooserTest, TreePhaseEntersRolloutOnUnvisitedChild)
{
    expect_tree_frame();
    ON_CALL(get_in_rollout, get_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_dispatches, get_dispatches(-1)).WillByDefault(Return(0));
    ON_CALL(compute_batch_size, compute_batch_size(0)).WillByDefault(Return(5));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(0));

    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(set_dispatches, set_dispatches(-1, 1));
    EXPECT_CALL(forestep, forestep(_));
    EXPECT_CALL(set_in_rollout, set_in_rollout(true)).Times(1);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

// ---------------------------------------------------------------------------
// UniformValueUpdateTest
// ---------------------------------------------------------------------------
class UniformValueUpdateTest : public ::testing::Test
{
protected:
    NiceMock<MockGetValue>      get_value;
    StrictMock<MockSetValue>    set_value;
    NiceMock<MockGetValueDelta> value_delta;
    monte_carlo::uniform_value_update<int,
                                      MockGetValue,
                                      MockSetValue,
                                      MockGetValueDelta> sut{
        get_value, set_value, value_delta};
};

TEST_F(UniformValueUpdateTest, UpdateAddsDeltaToCurrentValue)
{
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(1.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(0.5));
    EXPECT_CALL(set_value, set_value(42, 1.5));

    sut.update(42);
}

// ---------------------------------------------------------------------------
// Ucb1Test
// ---------------------------------------------------------------------------
struct stub_choice_count
{
    size_t size() const { return 2; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i); }
};

struct MockUcbWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

struct MockGetExplorationConstant
{
    MOCK_METHOD(double, get_exploration_constant, (const int&), (const));
};

class Ucb1Test : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits>               get_visits;
    NiceMock<MockGetValue>                get_value;
    NiceMock<MockUcbWalker>               walker;
    monte_carlo::uniform_exploration_constant<double> exploration{0.0};
    monte_carlo::ucb1<int, int, double,
                        MockGetVisits,
                        MockGetValue,
                        MockUcbWalker,
                        monte_carlo::uniform_exploration_constant<double>,
                        stub_choice_count,
                        stub_choice_at> sut{
        get_visits, get_value, walker, exploration};
};

TEST_F(Ucb1Test, PicksHighestValuePerVisitRatio)
{
    const int parent = 99;
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(10));
    ON_CALL(walker, walk(parent, 0)).WillByDefault(Return(0));
    ON_CALL(walker, walk(parent, 1)).WillByDefault(Return(1));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(3));
    ON_CALL(get_visits, get_visits(1)).WillByDefault(Return(2));
    ON_CALL(get_value, get_value(0)).WillByDefault(Return(6.0));
    ON_CALL(get_value, get_value(1)).WillByDefault(Return(10.0));

    stub_choice_count count;
    stub_choice_at    at;

    EXPECT_EQ(sut.policy_choose(parent, count, at), 1);
}

// ---------------------------------------------------------------------------
// SimCursorTest
// ---------------------------------------------------------------------------
class SimCursorTest : public ::testing::Test
{
protected:
    monte_carlo::sim_cursor<int> sut{-1};
};

TEST_F(SimCursorTest, StartsAtRoot)
{
    EXPECT_EQ(sut.get_current_node(), -1);
}

TEST_F(SimCursorTest, SetMovesCursor)
{
    sut.set_current_node(4);
    EXPECT_EQ(sut.get_current_node(), 4);
}

// ---------------------------------------------------------------------------
// SimBackpropPathTest
// ---------------------------------------------------------------------------
class SimBackpropPathTest : public ::testing::Test
{
protected:
    monte_carlo::sim_backprop_path<int> sut{-1};
};

TEST_F(SimBackpropPathTest, SeededWithRoot)
{
    EXPECT_EQ(sut.size(), 1u);
    EXPECT_EQ(sut.top(), -1);
}

TEST_F(SimBackpropPathTest, PushThenPopUnwindsInReverseOrder)
{
    sut.push(0);
    sut.push(1);

    EXPECT_EQ(sut.size(), 3u);

    EXPECT_EQ(sut.top(), 1);
    sut.pop();
    EXPECT_EQ(sut.top(), 0);
    sut.pop();
    EXPECT_EQ(sut.top(), -1);
    sut.pop();

    EXPECT_EQ(sut.size(), 0u);
}

// ---------------------------------------------------------------------------
// SimVisitCreditorTest
// ---------------------------------------------------------------------------
class SimVisitCreditorTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits>   get_visits;
    StrictMock<MockSetVisits> set_visits;
    monte_carlo::sim_visit_creditor<int,
                                    MockGetVisits,
                                    MockSetVisits> sut{get_visits, set_visits};
};

TEST_F(SimVisitCreditorTest, CreditWritesBackOneMoreVisit)
{
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(4));
    EXPECT_CALL(set_visits, set_visits(7, 5));

    sut.credit(7);
}

// ---------------------------------------------------------------------------
// SimValueCreditorTest
// ---------------------------------------------------------------------------
struct MockCreditVisitNode
{
    MOCK_METHOD(void, credit, (const int&), ());
};

struct MockUpdateNode
{
    MOCK_METHOD(void, update, (const int&), ());
};

class SimValueCreditorTest : public ::testing::Test
{
protected:
    StrictMock<MockCreditVisitNode> visit_creditor;
    StrictMock<MockUpdateNode>      update_node;
    monte_carlo::sim_value_creditor<int,
                                    MockCreditVisitNode,
                                    MockUpdateNode> sut{visit_creditor, update_node};
};

TEST_F(SimValueCreditorTest, CreditVisitsThenUpdatesValueForSameNode)
{
    InSequence seq;
    EXPECT_CALL(visit_creditor, credit(7));
    EXPECT_CALL(update_node, update(7));

    sut.credit(7);
}

// ---------------------------------------------------------------------------
// SimChooserTest
// ---------------------------------------------------------------------------
struct MockGetCurrentNode
{
    MOCK_METHOD(int, get_current_node, (), (const));
};

struct MockSetCurrentNode
{
    MOCK_METHOD(void, set_current_node, (int), ());
};

struct MockPushNode
{
    MOCK_METHOD(void, push, (const int&), ());
};

class SimChooserTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits>             get_visits;
    position_walker                     walker;
    StrictMock<MockPolicyChoose>        policy;
    StrictMock<MockRolloutChoose>       rollout;
    NiceMock<MockGetCurrentNode>        get_current_node;
    StrictMock<MockSetCurrentNode>      set_current_node;
    StrictMock<MockPushNode>            push_node;
    NiceMock<MockGetInRollout>          get_in_rollout;
    StrictMock<MockSetInRolloutChooser> set_in_rollout;
    std::vector<jump_t>                 jumps_{1};

    monte_carlo::sim_chooser<int, jump_t,
                             MockGetVisits,
                             position_walker,
                             std::vector<jump_t>,
                             std::vector<jump_t>,
                             MockPolicyChoose,
                             MockRolloutChoose,
                             MockGetCurrentNode,
                             MockSetCurrentNode,
                             MockPushNode,
                             MockGetInRollout,
                             MockSetInRolloutChooser> sut{
        get_visits,
        walker,
        policy,
        rollout,
        get_current_node,
        set_current_node,
        push_node,
        get_in_rollout,
        set_in_rollout};
};

TEST_F(SimChooserTest, RolloutPhaseAdvancesCursorWithoutPushingPath)
{
    ON_CALL(get_in_rollout, get_in_rollout()).WillByDefault(Return(true));
    ON_CALL(get_current_node, get_current_node()).WillByDefault(Return(2));

    EXPECT_CALL(rollout, rollout_choose(jumps_, jumps_)).WillOnce(Return(jump_t{1}));
    EXPECT_CALL(set_current_node, set_current_node(3));
    EXPECT_CALL(push_node, push(_)).Times(0);
    EXPECT_CALL(policy, policy_choose(_, _, _)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(SimChooserTest, TreePhasePushesChildAndAdvancesCursor)
{
    ON_CALL(get_in_rollout, get_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_current_node, get_current_node()).WillByDefault(Return(-1));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(5));

    InSequence seq;
    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(push_node, push(0));
    EXPECT_CALL(set_current_node, set_current_node(0));
    EXPECT_CALL(set_in_rollout, set_in_rollout(true)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(SimChooserTest, TreePhaseEntersRolloutOnUnvisitedChild)
{
    ON_CALL(get_in_rollout, get_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_current_node, get_current_node()).WillByDefault(Return(-1));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(0));

    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(push_node, push(0));
    EXPECT_CALL(set_current_node, set_current_node(0));
    EXPECT_CALL(set_in_rollout, set_in_rollout(true)).Times(1);
    EXPECT_CALL(rollout, rollout_choose(_, _)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

// ---------------------------------------------------------------------------
// SimTerminatorTest
// ---------------------------------------------------------------------------
struct MockGetNodeCount
{
    MOCK_METHOD(size_t, size, (), (const));
};

struct MockGetTopNode
{
    MOCK_METHOD(int, top, (), (const));
};

struct MockPopNode
{
    MOCK_METHOD(void, pop, (), ());
};

struct MockCreditNode
{
    MOCK_METHOD(void, credit, (const int&), ());
};

class SimTerminatorTest : public ::testing::Test
{
protected:
    StrictMock<MockGetNodeCount>   get_node_count;
    StrictMock<MockGetTopNode>     get_top_node;
    StrictMock<MockPopNode>        pop_node;
    StrictMock<MockCreditNode>     credit_node;
    StrictMock<MockSetInRollout>   set_in_rollout;
    monte_carlo::sim_terminator<int,
                                MockGetNodeCount,
                                MockGetTopNode,
                                MockPopNode,
                                MockCreditNode,
                                MockSetInRollout> sut{
        get_node_count, get_top_node, pop_node, credit_node, set_in_rollout};
};

TEST_F(SimTerminatorTest, TerminateDrainsPathLeafFirstThenClearsFlag)
{
    InSequence seq;
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(2));
    EXPECT_CALL(get_top_node, top()).WillOnce(Return(7));
    EXPECT_CALL(credit_node, credit(7));
    EXPECT_CALL(pop_node, pop());
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(1));
    EXPECT_CALL(get_top_node, top()).WillOnce(Return(3));
    EXPECT_CALL(credit_node, credit(3));
    EXPECT_CALL(pop_node, pop());
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(0));
    EXPECT_CALL(set_in_rollout, set_in_rollout(false));

    sut.terminate();
}

TEST_F(SimTerminatorTest, TerminateOnEmptyPathOnlyClearsFlag)
{
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(0));
    EXPECT_CALL(credit_node, credit(_)).Times(0);
    EXPECT_CALL(pop_node, pop()).Times(0);
    EXPECT_CALL(set_in_rollout, set_in_rollout(false));

    sut.terminate();
}

// ---------------------------------------------------------------------------
// SimInRolloutTest
// ---------------------------------------------------------------------------
class SimInRolloutTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::sim_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;
};

TEST_F(SimInRolloutTest, FlagTransitionsEpisodes1And2)
{
    const std::vector<double> track = {5.0};
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    {
        manifest_t m(visits, visits, value, value, rng, 1.0, -1);

        EXPECT_FALSE(m.in_rollout.get_in_rollout());

        m.chooser.choose(jumps, jumps);
        EXPECT_TRUE(m.in_rollout.get_in_rollout());

        m.chooser.choose(jumps, jumps);
        EXPECT_TRUE(m.in_rollout.get_in_rollout());

        m.delta.set_value(5.0);
        m.terminator.terminate();
        EXPECT_FALSE(m.in_rollout.get_in_rollout());
    }

    {
        manifest_t m(visits, visits, value, value, rng, 1.0, -1);

        EXPECT_FALSE(m.in_rollout.get_in_rollout());

        m.chooser.choose(jumps, jumps);
        EXPECT_FALSE(m.in_rollout.get_in_rollout());

        m.chooser.choose(jumps, jumps);
        EXPECT_TRUE(m.in_rollout.get_in_rollout());

        m.delta.set_value(5.0);
        m.terminator.terminate();
        EXPECT_FALSE(m.in_rollout.get_in_rollout());
    }
}

// ---------------------------------------------------------------------------
// SimTerminateBackpropTest
// ---------------------------------------------------------------------------
class SimTerminateBackpropTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t    = monte_carlo::value_table<int, double, std::unordered_map>;
    using manifest_t = monte_carlo::sim_value_manifest<
                          int, jump_t, double,
                          visits_t, visits_t, value_t, value_t,
                          position_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;

    void run_terminal_episode(manifest_t& m, const std::vector<jump_t>& jumps)
    {
        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (next >= 3)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                return;
            }
            position = next;
            reward   = static_cast<double>(next + 1);
        }
    }
};

TEST_F(SimTerminateBackpropTest, CreditsEveryNodeOnBackpropPath)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, -1);

    run_terminal_episode(m, jumps);

    EXPECT_EQ(visits.get_visits(-1), 1u);
    EXPECT_EQ(visits.get_visits(0), 1u);
    EXPECT_DOUBLE_EQ(value.get_value(-1), 3.0);
    EXPECT_DOUBLE_EQ(value.get_value(0), 3.0);
}

// ---------------------------------------------------------------------------
// DbuctStackLockstepTest
// ---------------------------------------------------------------------------
class DbuctStackLockstepTest : public ::testing::Test
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

    void run_episode(manifest_t& m, const std::vector<jump_t>& jumps)
    {
        int    position = -1;
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.get_in_rollout())
                EXPECT_EQ(m.frame_stack.top().handle, m.value_stack.top().handle);
            if (next >= 1)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                return;
            }
            position = next;
            reward   = 1.0;
        }
    }
};

TEST_F(DbuctStackLockstepTest, FrameAndValueStacksShareTopHandle)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 2, -1);

    run_episode(m, jumps);
    run_episode(m, jumps);
    run_episode(m, jumps);

    EXPECT_EQ(m.frame_stack.size(), 2u);
    EXPECT_EQ(m.frame_stack.top().handle, 0);
    EXPECT_EQ(m.value_stack.top().handle, 0);
}

// ---------------------------------------------------------------------------
// DbuctSingleEpisodeCreditTest
// ---------------------------------------------------------------------------
class DbuctSingleEpisodeCreditTest : public ::testing::Test
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
};

TEST_F(DbuctSingleEpisodeCreditTest, SeedEpisodeCreditsExpansionNodeAndRoot)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 1.0,
                 std::numeric_limits<size_t>::max(), -1);

    m.chooser.choose(jumps, jumps);
    m.chooser.choose(jumps, jumps);
    m.delta.set_value(8.0);
    m.terminator.terminate();

    EXPECT_EQ(visits.get_visits(-1), 1u);
    EXPECT_EQ(visits.get_visits(0), 1u);
    EXPECT_DOUBLE_EQ(value.get_value(-1), 8.0);
    EXPECT_DOUBLE_EQ(value.get_value(0), 8.0);
}

// ---------------------------------------------------------------------------
// DbuctManualBackstepValueLumpTest
// ---------------------------------------------------------------------------
class DbuctManualBackstepValueLumpTest : public ::testing::Test
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

    void run_episode(manifest_t& m, const std::vector<jump_t>& jumps, std::vector<int>& path)
    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.get_in_rollout())
                path.push_back(next);
            if (next >= 1)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                path.resize(m.frame_stack.size());
                return;
            }
            position = next;
            reward   = 1.0;
        }
    }
};

TEST_F(DbuctManualBackstepValueLumpTest, ManualBackstepRollsValueLumpIntoRoot)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(42);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 0.0, 2, -1);
    std::vector<int> path = {-1};

    run_episode(m, jumps, path);
    run_episode(m, jumps, path);

    const double root_value_before = value.get_value(-1);

    {
        int    position = path.back();
        double reward   = 0.0;

        while (true)
        {
            jump_t chosen = m.chooser.choose(jumps, jumps);
            int    next   = position + chosen;
            if (!m.in_rollout.get_in_rollout())
                path.push_back(next);
            if (next >= 1)
            {
                m.delta.set_value(reward);
                m.terminator.terminate();
                while (m.frame_stack.size() > 1)
                    m.value_stack_controller.backstep();
                path.resize(m.frame_stack.size());
                break;
            }
            position = next;
            reward   = 1.0;
        }
    }

    EXPECT_EQ(m.frame_stack.size(), 1u);
    EXPECT_DOUBLE_EQ(value.get_value(-1) - root_value_before, 1.0);
}
