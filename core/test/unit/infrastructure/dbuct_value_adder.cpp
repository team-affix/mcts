// dbuct_value_adder writes a value increment through to the bank and mirrors it
// into the top value frame's value lump.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_adder.hpp"
#include "value_objects/dbuct_value_frame.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

namespace
{

struct MockGetTopValueFrame
{
    MOCK_METHOD((monte_carlo::dbuct_value_frame<int, double>&), top, (), ());
};

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockSetValue
{
    MOCK_METHOD(void, set_value, (const int&, double), ());
};

}

class DbuctValueAdderTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_value_frame<int, double> value_frame_{3};
    NiceMock<MockGetTopValueFrame>              get_top_value_frame;
    NiceMock<MockGetValue>                      get_value;
    StrictMock<MockSetValue>                    set_value;
    monte_carlo::dbuct_value_adder<int, double,
                                   MockGetTopValueFrame,
                                   MockGetValue,
                                   MockSetValue> sut{get_top_value_frame, get_value, set_value};
};

TEST_F(DbuctValueAdderTest, AddValueIncrementsBankAndFrameLump)
{
    value_frame_.value_lump = 1.0;
    ON_CALL(get_top_value_frame, top()).WillByDefault(ReturnRef(value_frame_));
    ON_CALL(get_value, get_value(3)).WillByDefault(Return(4.0));
    EXPECT_CALL(set_value, set_value(3, 6.5));

    sut.add_value(2.5);

    EXPECT_DOUBLE_EQ(value_frame_.value_lump, 3.5);
}
