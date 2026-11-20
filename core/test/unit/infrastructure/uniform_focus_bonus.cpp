// uniform_focus_bonus gives the same bonus at every node.

#include <gtest/gtest.h>
#include "infrastructure/uniform_focus_bonus.hpp"

TEST(UniformFocusBonusTest, EveryNodeGetsTheSameBonus)
{
    const monte_carlo::uniform_focus_bonus<double> bonus(0.75);
    EXPECT_DOUBLE_EQ(bonus.get_focus_bonus(3), 0.75);
    EXPECT_DOUBLE_EQ(bonus.get_focus_bonus(-1), 0.75);
}
