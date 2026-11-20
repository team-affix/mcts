// episode_choice_log keeps an episode's choices in order until it is cleared.

#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/episode_choice_log.hpp"

class EpisodeChoiceLogTest : public ::testing::Test
{
protected:
    monte_carlo::episode_choice_log<int> log;
};

TEST_F(EpisodeChoiceLogTest, ALogStartsEmpty)
{
    EXPECT_TRUE(log.choices().empty());
}

TEST_F(EpisodeChoiceLogTest, ChoicesAreKeptInTheOrderPushed)
{
    log.push_choice(1);
    log.push_choice(0);
    log.push_choice(1);
    EXPECT_EQ(log.choices(), (std::vector<int>{1, 0, 1}));
}

TEST_F(EpisodeChoiceLogTest, ClearingEmptiesTheLog)
{
    log.push_choice(1);
    log.clear_choices();
    EXPECT_TRUE(log.choices().empty());
}

TEST_F(EpisodeChoiceLogTest, ChoicesPushedAfterAClearStartAfresh)
{
    log.push_choice(1);
    log.clear_choices();
    log.push_choice(0);
    EXPECT_EQ(log.choices(), (std::vector<int>{0}));
}
