// best_path_store keeps the best episode's choice at each node it passed
// through and that episode's reward; starting a new path replaces the old one.

#include <limits>
#include <optional>
#include <unordered_map>
#include <gtest/gtest.h>
#include "infrastructure/best_path_store.hpp"

class BestPathStoreTest : public ::testing::Test
{
protected:
    monte_carlo::best_path_store<int, int, double, std::unordered_map> store;
};

TEST_F(BestPathStoreTest, BeforeAnyPathTheBestRewardIsMinusInfinity)
{
    EXPECT_EQ(store.best_reward(), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(store.find_step(0), std::nullopt);
}

TEST_F(BestPathStoreTest, AStartedPathKeepsItsRewardAndSteps)
{
    store.start_path(-7.0);
    store.add_step(0, 1);
    store.add_step(3, 0);
    EXPECT_EQ(store.best_reward(), -7.0);
    EXPECT_EQ(store.find_step(0), std::optional<int>(1));
    EXPECT_EQ(store.find_step(3), std::optional<int>(0));
    EXPECT_EQ(store.find_step(5), std::nullopt);
}

TEST_F(BestPathStoreTest, StartingANewPathForgetsTheOldOnesSteps)
{
    store.start_path(-7.0);
    store.add_step(0, 1);
    store.add_step(3, 0);
    store.start_path(-4.0);
    store.add_step(0, 0);
    EXPECT_EQ(store.best_reward(), -4.0);
    EXPECT_EQ(store.find_step(0), std::optional<int>(0));
    EXPECT_EQ(store.find_step(3), std::nullopt);
}
