// The very first episode expands the root and credits both the expansion node
// and the root with the reward supplied to terminate().

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

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
                          std::mt19937>;
};

TEST_F(DbuctSingleEpisodeCreditTest, SeedEpisodeCreditsExpansionNodeAndRoot)
{
    const std::vector<jump_t> jumps = {1};
    std::mt19937              rng(0);
    visits_t                  visits;
    value_t                   value;

    manifest_t m(visits, visits, value, value, rng, 1.0, 0.0, -1);

    m.chooser.choose(jumps, jumps);
    m.chooser.choose(jumps, jumps);
    m.delta.set_value(8.0);
    m.terminator.terminate();

    EXPECT_EQ(visits.get_visits(-1), 1u);
    EXPECT_EQ(visits.get_visits(0), 1u);
    EXPECT_DOUBLE_EQ(value.get_value(-1), 8.0);
    EXPECT_DOUBLE_EQ(value.get_value(0), 8.0);
}
