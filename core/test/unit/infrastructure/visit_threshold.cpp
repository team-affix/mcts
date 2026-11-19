// visit_threshold is reached once the visits since the frontier became the
// frontier reach the threshold.

#include <gtest/gtest.h>
#include "infrastructure/visit_threshold.hpp"

TEST(VisitThresholdTest, BelowTheThresholdIsNotReached)
{
    EXPECT_FALSE(monte_carlo::visit_threshold(100).reached(99));
}

TEST(VisitThresholdTest, TheThresholdItselfIsReached)
{
    EXPECT_TRUE(monte_carlo::visit_threshold(100).reached(100));
}

TEST(VisitThresholdTest, AboveTheThresholdIsReached)
{
    EXPECT_TRUE(monte_carlo::visit_threshold(100).reached(5000));
}

TEST(VisitThresholdTest, AZeroThresholdIsAlwaysReached)
{
    EXPECT_TRUE(monte_carlo::visit_threshold(0).reached(0));
}
