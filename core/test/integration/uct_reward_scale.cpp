// With normalized ucb1, UCT's choices do not depend on the rewards' scale or
// offset: scaling every reward by a positive factor, or shifting every reward by
// a constant, leaves every node's visit count unchanged, so one c serves any
// reward scale.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_bounds.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class UctRewardScaleIntegrationTest : public ::testing::Test
{
protected:
    using visits_t   = monte_carlo::visits_table<std::vector<int>, path_unordered_map>;
    using value_t    = monte_carlo::value_table<std::vector<int>, double, path_unordered_map>;
    using bounds_t   = monte_carlo::value_bounds<double>;
    using manifest_t = monte_carlo::uct_value_manifest<
                          std::vector<int>, jump_t, double,
                          visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                          path_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;

    static constexpr int    kEpisodes = 300;
    static constexpr double kC        = 1.0;

    const std::vector<double> track{3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, 6.0};
    const std::vector<jump_t> jumps{1, 2, 3};

    // Trains on the coin-collecting track with every episode's total score
    // mapped to scale * score + offset.
    void train(visits_t& visits, value_t& value, bounds_t& bounds, double scale, double offset)
    {
        std::mt19937 rng(11);
        manifest_t   m(visits, visits, value, value, bounds, bounds, rng, kC, std::vector<int>{-1});
        for (int episode = 0; episode < kEpisodes; ++episode)
        {
            int    position = -1;
            double score    = 0.0;
            while (true)
            {
                position += m.chooser.choose(jumps, jumps);
                if (position >= static_cast<int>(track.size()))
                    break;
                score += track[position];
            }
            m.delta.set_value(scale * score + offset);
            m.terminator.terminate();
        }
    }

    // Every route of up to three moves from the root.
    std::vector<std::vector<int>> shallow_nodes() const
    {
        std::vector<std::vector<int>> nodes{{-1}};
        for (size_t start = 0; start < nodes.size(); ++start)
        {
            if (nodes[start].size() > 3)
                continue;
            for (jump_t jump : jumps)
                nodes.push_back(path_walker().walk(nodes[start], jump));
        }
        return nodes;
    }

    void expect_same_visits(const visits_t& expected, const visits_t& actual) const
    {
        for (const std::vector<int>& node : shallow_nodes())
            EXPECT_EQ(expected.get_visits(node), actual.get_visits(node));
    }

    visits_t plain_visits;
    value_t  plain_value;
    bounds_t plain_bounds;
};

TEST_F(UctRewardScaleIntegrationTest, ScalingEveryRewardChangesNoChoice)
{
    visits_t scaled_visits;
    value_t  scaled_value;
    bounds_t scaled_bounds;
    train(plain_visits, plain_value, plain_bounds, 1.0, 0.0);
    train(scaled_visits, scaled_value, scaled_bounds, 1024.0, 0.0);

    expect_same_visits(plain_visits, scaled_visits);
}

TEST_F(UctRewardScaleIntegrationTest, ShiftingEveryRewardChangesNoChoice)
{
    visits_t shifted_visits;
    value_t  shifted_value;
    bounds_t shifted_bounds;
    train(plain_visits, plain_value, plain_bounds, 1.0, 0.0);
    train(shifted_visits, shifted_value, shifted_bounds, 1.0, -4096.0);

    expect_same_visits(plain_visits, shifted_visits);
}

TEST_F(UctRewardScaleIntegrationTest, TheSearchExploresBeyondTheRoot)
{
    train(plain_visits, plain_value, plain_bounds, 1.0, 0.0);

    size_t visited = 0;
    for (const std::vector<int>& node : shallow_nodes())
        visited += plain_visits.get_visits(node) > 0;
    EXPECT_GT(visited, 20u);
}
