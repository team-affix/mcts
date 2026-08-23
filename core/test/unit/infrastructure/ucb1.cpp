// ucb1 with a zero exploration constant reduces to pure exploitation: it picks
// the child with the highest value-per-visit ratio.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/ucb1.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"

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

}

class Ucb1Test : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits>               get_visits;
    NiceMock<MockGetValue>                get_value;
    NiceMock<MockUcbWalker>               walker;
    monte_carlo::uniform_exploration_constant<double> exploration{0.0};
    monte_carlo::ucb1<int, int, double,
                        MockGetVisits,
                        MockGetValue,
                        MockUcbWalker,
                        monte_carlo::uniform_exploration_constant<double>,
                        stub_choice_count,
                        stub_choice_at> sut{
        get_visits, get_value, walker, exploration};
};

TEST_F(Ucb1Test, PicksHighestValuePerVisitRatio)
{
    const int parent = 99;
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(10));
    ON_CALL(walker, walk(parent, 0)).WillByDefault(Return(0));
    ON_CALL(walker, walk(parent, 1)).WillByDefault(Return(1));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(3));
    ON_CALL(get_visits, get_visits(1)).WillByDefault(Return(2));
    ON_CALL(get_value, get_value(0)).WillByDefault(Return(6.0));
    ON_CALL(get_value, get_value(1)).WillByDefault(Return(10.0));

    stub_choice_count count;
    stub_choice_at    at;

    EXPECT_EQ(sut.policy_choose(parent, count, at), 1);
}
