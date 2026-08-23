// uniform_value_update adds the node's delta onto its current banked value.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uniform_value_update.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace
{

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

class UniformValueUpdateTest : public ::testing::Test
{
protected:
    NiceMock<MockGetValue>      get_value;
    StrictMock<MockSetValue>    set_value;
    NiceMock<MockGetValueDelta> value_delta;
    monte_carlo::uniform_value_update<int,
                                      MockGetValue,
                                      MockSetValue,
                                      MockGetValueDelta> sut{
        get_value, set_value, value_delta};
};

TEST_F(UniformValueUpdateTest, UpdateAddsDeltaToCurrentValue)
{
    ON_CALL(get_value, get_value(42)).WillByDefault(Return(1.0));
    ON_CALL(value_delta, get_value_delta(42)).WillByDefault(Return(0.5));
    EXPECT_CALL(set_value, set_value(42, 1.5));

    sut.update(42);
}
