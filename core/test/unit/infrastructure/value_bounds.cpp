// value_bounds keeps the lowest and highest values included and maps a value
// onto [0, 1] between them; with no spread yet every value maps to 0.

#include <gtest/gtest.h>
#include "infrastructure/value_bounds.hpp"

class ValueBoundsTest : public ::testing::Test
{
protected:
    monte_carlo::value_bounds<double> bounds;
};

TEST_F(ValueBoundsTest, WithNothingIncludedEveryValueNormalizesToZero)
{
    EXPECT_DOUBLE_EQ(bounds.normalize(-7.0), 0.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(3.0), 0.0);
}

TEST_F(ValueBoundsTest, OneValueGivesNoSpread)
{
    bounds.include(-7.0);
    bounds.include(-7.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(-7.0), 0.0);
}

TEST_F(ValueBoundsTest, TheLowestMapsToZeroAndTheHighestToOne)
{
    bounds.include(-2000.0);
    bounds.include(-500.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(-2000.0), 0.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(-500.0), 1.0);
}

TEST_F(ValueBoundsTest, ValuesBetweenMapLinearly)
{
    bounds.include(-2000.0);
    bounds.include(-500.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(-1625.0), 0.25);
}

TEST_F(ValueBoundsTest, TheOrderOfInclusionDoesNotMatter)
{
    bounds.include(-500.0);
    bounds.include(-1250.0);
    bounds.include(-2000.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(-1250.0), 0.5);
}

TEST_F(ValueBoundsTest, BoundsOnlyWiden)
{
    bounds.include(0.0);
    bounds.include(10.0);
    bounds.include(5.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(5.0), 0.5);
    bounds.include(-10.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(5.0), 0.75);
}

TEST_F(ValueBoundsTest, AValueOutsideTheBoundsMapsOutsideTheUnitRange)
{
    bounds.include(0.0);
    bounds.include(10.0);
    EXPECT_DOUBLE_EQ(bounds.normalize(15.0), 1.5);
    EXPECT_DOUBLE_EQ(bounds.normalize(-5.0), -0.5);
}
