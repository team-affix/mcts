// power_band is A * N^(-r): A at one visit, shrinking as a power of the
// node's visits.

#include <gtest/gtest.h>
#include "infrastructure/power_band.hpp"

TEST(PowerBandTest, OneVisitGivesTheStartingWidth)
{
    EXPECT_DOUBLE_EQ(monte_carlo::power_band<double>(0.1, 0.5).get_band(1), 0.1);
}

TEST(PowerBandTest, TheBandShrinksAsAPowerOfTheVisits)
{
    EXPECT_DOUBLE_EQ(monte_carlo::power_band<double>(0.1, 0.5).get_band(100), 0.01);
    EXPECT_DOUBLE_EQ(monte_carlo::power_band<double>(0.2, 1.0).get_band(4), 0.05);
}

TEST(PowerBandTest, ASmallerRateShrinksMoreSlowly)
{
    const double slow = monte_carlo::power_band<double>(0.1, 0.1).get_band(100000);
    const double fast = monte_carlo::power_band<double>(0.1, 0.5).get_band(100000);
    EXPECT_GT(slow, fast);
    EXPECT_NEAR(slow, 0.1 * std::pow(100000.0, -0.1), 1e-12);
}

TEST(PowerBandTest, MoreVisitsNeverWidenTheBand)
{
    const monte_carlo::power_band<double> band(0.1, 0.1);
    for (size_t visits = 1; visits < 1000; ++visits)
        EXPECT_LE(band.get_band(visits + 1), band.get_band(visits));
}
