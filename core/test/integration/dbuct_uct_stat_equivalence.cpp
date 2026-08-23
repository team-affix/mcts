// With GII=SIZE_MAX, dbuct must produce bit-identical bank stats to vanilla uct
// after N episodes driven from the same RNG state.  All assertions operate
// exclusively on caller-owned bank queries: get_visits() and get_value().

#include <limits>
#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class DbuctStatEquivalenceTest : public ::testing::Test
{
protected:
    using visits_t        = monte_carlo::visits_table<int, std::unordered_map>;
    using value_t         = monte_carlo::value_table<int, double, std::unordered_map>;
    using uct_value_manifest_t = monte_carlo::uct_value_manifest<
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

    // One uct episode using the terminal-reward convention (reward = last
    // in-bounds position's track value).
    void uct_episode(visits_t&                  visits,
                     value_t&                   value,
                     const std::vector<double>& track,
                     const std::vector<jump_t>& jumps,
                     std::mt19937&              rng,
                     double                     c)
    {
        uct_value_manifest_t m(visits, visits, value, value, rng, c, -1);

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
};

TEST_F(DbuctStatEquivalenceTest, MatchesUctSeed100Track5Moves12)
{
    const std::vector<double> track = {3.0, 1.0, 4.0, 1.0, 5.0};
    const std::vector<jump_t> jumps = {1, 2};
    const double              c     = 5.0;
    const int                 N     = 200;

    std::mt19937 rng1(100), rng2(100);
    visits_t     uct_visits,   dbuct_visits;
    value_t      uct_value,    dbuct_value;

    for (int i = 0; i < N; ++i)
        uct_episode(uct_visits, uct_value, track, jumps, rng1, c);
    dbuct_episodes(dbuct_visits, dbuct_value, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(uct_visits.get_visits(pos), dbuct_visits.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(uct_value.get_value(pos), dbuct_value.get_value(pos))
            << "value mismatch at pos=" << pos;
    }
}

TEST_F(DbuctStatEquivalenceTest, MatchesUctSeed200Track4Moves13)
{
    const std::vector<double> track = {2.0, 7.0, 1.0, 8.0};
    const std::vector<jump_t> jumps = {1, 3};
    const double              c     = 8.0;
    const int                 N     = 300;

    std::mt19937 rng1(200), rng2(200);
    visits_t     uct_visits,   dbuct_visits;
    value_t      uct_value,    dbuct_value;

    for (int i = 0; i < N; ++i)
        uct_episode(uct_visits, uct_value, track, jumps, rng1, c);
    dbuct_episodes(dbuct_visits, dbuct_value, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(uct_visits.get_visits(pos), dbuct_visits.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(uct_value.get_value(pos), dbuct_value.get_value(pos))
            << "value mismatch at pos=" << pos;
    }
}

TEST_F(DbuctStatEquivalenceTest, MatchesUctSeed300Track6Moves123)
{
    const std::vector<double> track = {9.0, 2.0, 6.0, 5.0, 3.0, 5.0};
    const std::vector<jump_t> jumps = {1, 2, 3};
    const double              c     = 9.0;
    const int                 N     = 500;

    std::mt19937 rng1(300), rng2(300);
    visits_t     uct_visits,   dbuct_visits;
    value_t      uct_value,    dbuct_value;

    for (int i = 0; i < N; ++i)
        uct_episode(uct_visits, uct_value, track, jumps, rng1, c);
    dbuct_episodes(dbuct_visits, dbuct_value, track, jumps, rng2, c, N);

    for (int pos = -1; pos < static_cast<int>(track.size()); ++pos)
    {
        EXPECT_EQ(uct_visits.get_visits(pos), dbuct_visits.get_visits(pos))
            << "visits mismatch at pos=" << pos;
        EXPECT_DOUBLE_EQ(uct_value.get_value(pos), dbuct_value.get_value(pos))
            << "value mismatch at pos=" << pos;
    }
}
