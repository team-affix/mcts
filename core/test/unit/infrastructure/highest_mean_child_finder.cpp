// highest_mean_child_finder picks the choice leading to the visited child with
// the highest mean value, the first on a tie, ignoring unvisited children, and
// nothing while no child has a visit.

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/highest_mean_child_finder.hpp"

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

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

using finder_t = monte_carlo::highest_mean_child_finder<int, int, double, MockGetVisits, MockGetValue, MockWalker,
                                                        stub_choice_count, stub_choice_at>;

}

class HighestMeanChildFinderTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits> get_visits;
    NiceMock<MockGetValue>  get_value;
    NiceMock<MockWalker>    walker;
    finder_t                sut{get_visits, get_value, walker};
    stub_choice_count       count;
    stub_choice_at          at;

    void SetUp() override
    {
        for (int choice = 0; choice <= 20; choice += 10)
            ON_CALL(walker, walk(4, choice)).WillByDefault(Return(100 + choice));
    }

    void set_child(int child, size_t visits, double mean)
    {
        ON_CALL(get_visits, get_visits(child)).WillByDefault(Return(visits));
        ON_CALL(get_value, get_value(child)).WillByDefault(Return(mean * static_cast<double>(visits)));
    }
};

TEST_F(HighestMeanChildFinderTest, PicksTheHighestMeanNotTheMostVisits)
{
    set_child(100, 900, -2000.0);
    set_child(110, 40, -1500.0);
    set_child(120, 60, -1800.0);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(10));
}

TEST_F(HighestMeanChildFinderTest, NegativeMeansCompareCorrectly)
{
    set_child(100, 5, -3.0);
    set_child(110, 5, -7.0);
    set_child(120, 5, -9.0);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(0));
}

TEST_F(HighestMeanChildFinderTest, ATieGoesToTheFirstChild)
{
    set_child(100, 5, -9.0);
    set_child(110, 3, -2.0);
    set_child(120, 8, -2.0);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(10));
}

TEST_F(HighestMeanChildFinderTest, UnvisitedChildrenAreIgnored)
{
    set_child(100, 0, 0.0);
    set_child(110, 0, 0.0);
    set_child(120, 2, -50.0);
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(20));
}

TEST_F(HighestMeanChildFinderTest, WithNoVisitedChildNothingIsPicked)
{
    set_child(100, 0, 0.0);
    set_child(110, 0, 0.0);
    set_child(120, 0, 0.0);
    EXPECT_EQ(sut.find(4, count, at), std::nullopt);
}
