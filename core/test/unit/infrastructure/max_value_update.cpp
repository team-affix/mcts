// max_value_update keeps the best reward backed up through a node: the first
// visit takes the reward as is, later visits only raise it.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/max_value_update.hpp"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace
{

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockSetValue
{
    MOCK_METHOD(void, set_value, (const int&, double), ());
};

struct MockGetValueDelta
{
    MOCK_METHOD(double, get_value_delta, (const int&), (const));
};

}

class MaxValueUpdateTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits>     get_visits;
    NiceMock<MockGetValue>      get_value;
    StrictMock<MockSetValue>    set_value;
    NiceMock<MockGetValueDelta> value_delta;
    monte_carlo::max_value_update<int,
                                  MockGetVisits,
                                  MockGetValue,
                                  MockSetValue,
                                  MockGetValueDelta> sut{
        get_visits, get_value, set_value, value_delta};
};

TEST_F(MaxValueUpdateTest, FirstVisitTakesTheRewardEvenBelowTheUnseenValue)
{
    ON_CALL(get_visits, get_visits(42)).WillByDefault(Return(1));
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(0.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(-7.0));
    EXPECT_CALL(set_value, set_value(42, -7.0));

    sut.update(42);
}

TEST_F(MaxValueUpdateTest, ABetterRewardReplacesTheBest)
{
    ON_CALL(get_visits, get_visits(42)).WillByDefault(Return(5));
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(-7.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(-3.0));
    EXPECT_CALL(set_value, set_value(42, -3.0));

    sut.update(42);
}

TEST_F(MaxValueUpdateTest, AWorseRewardLeavesTheBest)
{
    ON_CALL(get_visits, get_visits(42)).WillByDefault(Return(5));
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(-3.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(-7.0));
    EXPECT_CALL(set_value, set_value(_, _)).Times(0);

    sut.update(42);
}

TEST_F(MaxValueUpdateTest, AnEqualRewardLeavesTheBest)
{
    ON_CALL(get_visits, get_visits(42)).WillByDefault(Return(2));
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(4.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(4.0));
    EXPECT_CALL(set_value, set_value(_, _)).Times(0);

    sut.update(42);
}

TEST_F(MaxValueUpdateTest, TheSecondVisitComparesWithTheFirstReward)
{
    ON_CALL(get_visits, get_visits(42)).WillByDefault(Return(2));
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(-7.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(-9.0));
    EXPECT_CALL(set_value, set_value(_, _)).Times(0);

    sut.update(42);
}

TEST_F(MaxValueUpdateTest, EachNodeIsUpdatedOnItsOwnValue)
{
    ON_CALL(get_visits, get_visits(_)).WillByDefault(Return(3));
    ON_CALL(get_value, get_value(1)).WillByDefault(Return(10.0));
    ON_CALL(get_value, get_value(2)).WillByDefault(Return(-10.0));
    ON_CALL(value_delta, get_value_delta(_)).WillByDefault(Return(0.0));
    EXPECT_CALL(set_value, set_value(2, 0.0));

    sut.update(1);
    sut.update(2);
}
