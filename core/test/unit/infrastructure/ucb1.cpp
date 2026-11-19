// ucb1 records each visited child's mean, then scores children by their
// normalized mean plus c * sqrt(ln(parent visits) / child visits). Unvisited
// children are chosen first.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/ucb1.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::InSequence;
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

struct MockIncludeValue
{
    MOCK_METHOD(void, include, (double), ());
};

struct MockNormalizeValue
{
    MOCK_METHOD(double, normalize, (double), (const));
};

struct MockUcbWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

using policy_t = monte_carlo::ucb1<int, int, double,
                                   MockGetVisits,
                                   MockGetValue,
                                   MockIncludeValue,
                                   MockNormalizeValue,
                                   MockUcbWalker,
                                   monte_carlo::uniform_exploration_constant<double>,
                                   stub_choice_count,
                                   stub_choice_at>;

}

class Ucb1Test : public ::testing::Test
{
protected:
    static constexpr int parent = 99;

    NiceMock<MockGetVisits>                           get_visits;
    NiceMock<MockGetValue>                            get_value;
    NiceMock<MockIncludeValue>                        include_value;
    NiceMock<MockNormalizeValue>                      normalize_value;
    NiceMock<MockUcbWalker>                           walker;
    monte_carlo::uniform_exploration_constant<double> no_exploration{0.0};
    monte_carlo::uniform_exploration_constant<double> unit_exploration{1.0};
    policy_t greedy{get_visits, get_value, include_value, normalize_value, walker, no_exploration};
    policy_t exploring{get_visits, get_value, include_value, normalize_value, walker, unit_exploration};
    stub_choice_count count;
    stub_choice_at    at;

    void SetUp() override
    {
        ON_CALL(walker, walk(parent, 0)).WillByDefault(Return(0));
        ON_CALL(walker, walk(parent, 1)).WillByDefault(Return(1));
    }

    // A child with the given visits and mean, whose mean normalizes to `normalized`.
    void set_child(int child, size_t visits, double mean, double normalized)
    {
        ON_CALL(get_visits, get_visits(child)).WillByDefault(Return(visits));
        ON_CALL(get_value, get_value(child)).WillByDefault(Return(mean * static_cast<double>(visits)));
        ON_CALL(normalize_value, normalize(mean)).WillByDefault(Return(normalized));
    }
};

TEST_F(Ucb1Test, GreedyPicksTheHighestNormalizedMean)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(5));
    set_child(0, 3, -2000.0, 0.2);
    set_child(1, 2, -600.0, 0.9);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}

TEST_F(Ucb1Test, ScoresUseTheNormalizedMeanNotTheRawOne)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(5));
    // Raw means would pick child 1; the normalizer says otherwise.
    set_child(0, 3, -2000.0, 0.9);
    set_child(1, 2, -600.0, 0.2);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
}

TEST_F(Ucb1Test, EveryVisitedChildsMeanIsRecorded)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(5));
    set_child(0, 3, -2000.0, 0.0);
    set_child(1, 2, -600.0, 1.0);

    EXPECT_CALL(include_value, include(_)).Times(AnyNumber());
    EXPECT_CALL(include_value, include(-2000.0));
    EXPECT_CALL(include_value, include(-600.0));
    greedy.policy_choose(parent, count, at);
}

TEST_F(Ucb1Test, AllMeansAreRecordedBeforeAnyIsNormalized)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(5));
    set_child(0, 3, -2000.0, 0.0);
    set_child(1, 2, -600.0, 1.0);

    InSequence order;
    EXPECT_CALL(include_value, include(-2000.0));
    EXPECT_CALL(include_value, include(-600.0));
    EXPECT_CALL(normalize_value, normalize(-2000.0)).WillOnce(Return(0.0));
    EXPECT_CALL(normalize_value, normalize(-600.0)).WillOnce(Return(1.0));
    greedy.policy_choose(parent, count, at);
}

TEST_F(Ucb1Test, AnUnvisitedChildIsChosenBeforeAnyVisitedOne)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(5));
    set_child(0, 5, 100.0, 1.0);
    set_child(1, 0, 0.0, 0.0);

    EXPECT_CALL(normalize_value, normalize(_)).Times(0);
    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}

TEST_F(Ucb1Test, TheFirstUnvisitedChildIsChosen)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(0));
    set_child(0, 0, 0.0, 0.0);
    set_child(1, 0, 0.0, 0.0);

    EXPECT_CALL(include_value, include(_)).Times(0);
    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
}

TEST_F(Ucb1Test, ExplorationFavoursTheLessVisitedChildAtEqualValue)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(11));
    set_child(0, 10, -5.0, 0.5);
    set_child(1, 1, -5.0, 0.5);

    EXPECT_EQ(exploring.policy_choose(parent, count, at), 1);
}

TEST_F(Ucb1Test, AnExplorationConstantAboveOneCanOutweighTheWholeScale)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(101));
    // Bonus: sqrt(ln 101 / 100) ~ 0.21 against sqrt(ln 101 / 1) ~ 2.15.
    set_child(0, 100, -500.0, 1.0);
    set_child(1, 1, -2000.0, 0.0);
    monte_carlo::uniform_exploration_constant<double> slight{0.4};
    monte_carlo::uniform_exploration_constant<double> heavy{2.0};
    policy_t slightly_exploring{get_visits, get_value, include_value, normalize_value, walker, slight};
    policy_t heavily_exploring{get_visits, get_value, include_value, normalize_value, walker, heavy};

    EXPECT_EQ(slightly_exploring.policy_choose(parent, count, at), 0);
    EXPECT_EQ(heavily_exploring.policy_choose(parent, count, at), 1);
}

TEST_F(Ucb1Test, ATieGoesToTheFirstChild)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(8));
    set_child(0, 4, -2.0, 0.5);
    set_child(1, 4, -2.0, 0.5);

    EXPECT_EQ(exploring.policy_choose(parent, count, at), 0);
}

TEST_F(Ucb1Test, ChildrenAreReachedThroughTheWalker)
{
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(8));
    ON_CALL(walker, walk(parent, 0)).WillByDefault(Return(7));
    ON_CALL(walker, walk(parent, 1)).WillByDefault(Return(3));
    set_child(7, 4, -9.0, 0.0);
    set_child(3, 4, -1.0, 1.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}
