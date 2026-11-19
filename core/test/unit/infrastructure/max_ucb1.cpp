// max_ucb1 scores a visited child by its best backed-up reward plus the UCB1
// exploration bonus; unvisited children are chosen first.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/max_ucb1.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct stub_choice_count
{
    size_t size() const { return 2; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i); }
};

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockUcbWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

using policy_t = monte_carlo::max_ucb1<int, int, double,
                                       MockGetVisits,
                                       MockGetValue,
                                       MockUcbWalker,
                                       monte_carlo::uniform_exploration_constant<double>,
                                       stub_choice_count,
                                       stub_choice_at>;

}

class MaxUcb1Test : public ::testing::Test
{
protected:
    static constexpr int parent = 99;

    NiceMock<MockGetVisits>                           get_visits;
    NiceMock<MockGetValue>                            get_value;
    NiceMock<MockUcbWalker>                           walker;
    monte_carlo::uniform_exploration_constant<double> no_exploration{0.0};
    monte_carlo::uniform_exploration_constant<double> unit_exploration{1.0};
    policy_t greedy{get_visits, get_value, walker, no_exploration};
    policy_t exploring{get_visits, get_value, walker, unit_exploration};
    stub_choice_count count;
    stub_choice_at    at;

    void SetUp() override
    {
        ON_CALL(walker, walk(parent, 0)).WillByDefault(Return(0));
        ON_CALL(walker, walk(parent, 1)).WillByDefault(Return(1));
    }

    void set_child(int child, size_t visits, double best)
    {
        ON_CALL(get_visits, get_visits(child)).WillByDefault(Return(visits));
        ON_CALL(get_value, get_value(child)).WillByDefault(Return(best));
    }
};

TEST_F(MaxUcb1Test, PicksTheHighestBestRewardNotTheHighestRatio)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(13));
    // Per visit, child 1 would win (-0.8 against -2); by best reward child 0 wins.
    set_child(0, 3, -6.0);
    set_child(1, 10, -8.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
}

TEST_F(MaxUcb1Test, AnUnvisitedChildIsChosenBeforeAnyVisitedOne)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(5));
    set_child(0, 5, 100.0);
    set_child(1, 0, 0.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}

TEST_F(MaxUcb1Test, TheFirstUnvisitedChildIsChosen)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(0));
    set_child(0, 0, 0.0);
    set_child(1, 0, 0.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
}

TEST_F(MaxUcb1Test, ExplorationFavoursTheLessVisitedChildAtEqualBest)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(11));
    set_child(0, 10, -5.0);
    set_child(1, 1, -5.0);

    EXPECT_EQ(exploring.policy_choose(parent, count, at), 1);
}

TEST_F(MaxUcb1Test, ExplorationCanOutweighASmallGapInBest)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(101));
    // Bonus: sqrt(ln 101 / 100) ~ 0.21 against sqrt(ln 101 / 1) ~ 2.15.
    set_child(0, 100, -5.0);
    set_child(1, 1, -6.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
    EXPECT_EQ(exploring.policy_choose(parent, count, at), 1);
}

TEST_F(MaxUcb1Test, ATieGoesToTheFirstChild)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(8));
    set_child(0, 4, -2.0);
    set_child(1, 4, -2.0);

    EXPECT_EQ(exploring.policy_choose(parent, count, at), 0);
}

TEST_F(MaxUcb1Test, ChildrenAreReachedThroughTheWalker)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(8));
    ON_CALL(walker, walk(parent, 0)).WillByDefault(Return(7));
    ON_CALL(walker, walk(parent, 1)).WillByDefault(Return(3));
    set_child(7, 4, -9.0);
    set_child(3, 4, -1.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}
