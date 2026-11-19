// most_visited_child_finder picks the choice leading to the most visited child,
// the first on a tie, and nothing while no child has a visit.

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/most_visited_child_finder.hpp"

using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct stub_choice_count
{
    size_t size() const { return 3; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i) * 10; }
};

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

using finder_t = monte_carlo::most_visited_child_finder<int, int, MockGetVisits, MockWalker,
                                                        stub_choice_count, stub_choice_at>;

}

class MostVisitedChildFinderTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits> get_visits;
    NiceMock<MockWalker>    walker;
    finder_t                sut{get_visits, walker};
    stub_choice_count       count;
    stub_choice_at          at;

    void SetUp() override
    {
        for (int choice = 0; choice <= 20; choice += 10)
            ON_CALL(walker, walk(4, choice)).WillByDefault(Return(100 + choice));
    }

    void set_child_visits(size_t first, size_t second, size_t third)
    {
        ON_CALL(get_visits, get_visits(100)).WillByDefault(Return(first));
        ON_CALL(get_visits, get_visits(110)).WillByDefault(Return(second));
        ON_CALL(get_visits, get_visits(120)).WillByDefault(Return(third));
    }
};

TEST_F(MostVisitedChildFinderTest, PicksTheMostVisitedChild)
{
    set_child_visits(30, 50, 20);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(10));
}

TEST_F(MostVisitedChildFinderTest, ATieGoesToTheFirstChild)
{
    set_child_visits(20, 50, 50);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(10));
}

TEST_F(MostVisitedChildFinderTest, TheLastChildCanBePicked)
{
    set_child_visits(1, 2, 3);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(20));
}

TEST_F(MostVisitedChildFinderTest, WithNoVisitedChildNothingIsPicked)
{
    set_child_visits(0, 0, 0);
    EXPECT_EQ(sut.find(4, count, at), std::nullopt);
}
