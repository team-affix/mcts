// rollout_choice_log keeps the rollout's choices in the order they were made
// until it is cleared.

#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/rollout_choice_log.hpp"

class RolloutChoiceLogTest : public ::testing::Test
{
protected:
    monte_carlo::rollout_choice_log<int> log;
};

TEST_F(RolloutChoiceLogTest, ALogStartsEmpty)
{
    EXPECT_TRUE(log.choices().empty());
}

TEST_F(RolloutChoiceLogTest, ChoicesAreKeptInTheOrderPushed)
{
    log.push_choice(3);
    log.push_choice(1);
    log.push_choice(3);
    EXPECT_EQ(log.choices(), (std::vector<int>{3, 1, 3}));
}

TEST_F(RolloutChoiceLogTest, ClearingEmptiesTheLog)
{
    log.push_choice(3);
    log.push_choice(1);
    log.clear_choices();
    EXPECT_TRUE(log.choices().empty());
}

TEST_F(RolloutChoiceLogTest, ChoicesPushedAfterAClearStartAfresh)
{
    log.push_choice(3);
    log.clear_choices();
    log.push_choice(5);
    EXPECT_EQ(log.choices(), (std::vector<int>{5}));
}

TEST_F(RolloutChoiceLogTest, ClearingAnEmptyLogLeavesItEmpty)
{
    log.clear_choices();
    EXPECT_TRUE(log.choices().empty());
}
