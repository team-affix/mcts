// Best-path memory through best_path_value_manifest: an episode whose reward is
// at least the best so far adds its whole path, rollout included, to the tree;
// a worse one adds only its first new node, as UCT does. Uses path_walker so
// every distinct route is its own node.

#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/best_path_value_manifest.hpp"
#include "infrastructure/value_bounds.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class BestPathMemoryIntegrationTest : public ::testing::Test
{
protected:
    using node_t     = std::vector<int>;
    using visits_t   = monte_carlo::visits_table<node_t, path_unordered_map>;
    using value_t    = monte_carlo::value_table<node_t, double, path_unordered_map>;
    using bounds_t   = monte_carlo::value_bounds<double>;
    using manifest_t = monte_carlo::best_path_value_manifest<
                          node_t, jump_t, double,
                          visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                          path_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;

    static constexpr size_t kDepth = 5;

    visits_t            visits;
    value_t             value;
    bounds_t            bounds;
    std::mt19937        rng{7};
    std::vector<jump_t> jumps{1, 2};
    manifest_t          m{visits, visits, value, value, bounds, bounds, rng, 1.0, node_t{0}};

    // Makes kDepth choices, then ends the episode with the given reward.
    // Returns the nodes the episode passed through below the root.
    std::vector<node_t> play(double reward)
    {
        std::vector<node_t> nodes;
        node_t              node{0};
        for (size_t step = 0; step < kDepth; ++step)
        {
            node = path_walker().walk(node, m.chooser.choose(jumps, jumps));
            nodes.push_back(node);
        }
        m.delta.set_value(reward);
        m.terminator.terminate();
        return nodes;
    }
};

TEST_F(BestPathMemoryIntegrationTest, TheFirstEpisodesWholePathJoinsTheTree)
{
    const std::vector<node_t> first = play(-3.0);

    for (const node_t& node : first)
    {
        EXPECT_EQ(visits.get_visits(node), 1u);
        EXPECT_DOUBLE_EQ(value.get_value(node), -3.0);
    }
}

TEST_F(BestPathMemoryIntegrationTest, AWorseEpisodeAddsOnlyItsFirstNewNode)
{
    play(-3.0);
    const std::vector<node_t> second = play(-5.0);

    EXPECT_EQ(visits.get_visits(second[0]), 1u);
    for (size_t step = 1; step < kDepth; ++step)
        EXPECT_EQ(visits.get_visits(second[step]), 0u);
}

TEST_F(BestPathMemoryIntegrationTest, AnEpisodeEqualToTheBestJoinsTheTreeWhole)
{
    play(-3.0);
    const std::vector<node_t> second = play(-3.0);

    for (const node_t& node : second)
        EXPECT_EQ(visits.get_visits(node), 1u);
}

TEST_F(BestPathMemoryIntegrationTest, ABetterEpisodeJoinsTheTreeWhole)
{
    play(-3.0);
    const std::vector<node_t> second = play(-1.0);

    for (const node_t& node : second)
        EXPECT_EQ(visits.get_visits(node), 1u);
}

TEST_F(BestPathMemoryIntegrationTest, AWorseEpisodeDoesNotLowerTheBar)
{
    play(-3.0);
    play(-5.0);
    const std::vector<node_t> third = play(-4.0);

    EXPECT_EQ(visits.get_visits(third[kDepth - 1]), 0u);
}

TEST_F(BestPathMemoryIntegrationTest, TheRootIsCreditedOncePerEpisode)
{
    play(-3.0);
    play(-5.0);
    play(-1.0);

    EXPECT_EQ(visits.get_visits(node_t{0}), 3u);
    EXPECT_DOUBLE_EQ(value.get_value(node_t{0}), -9.0);
}
