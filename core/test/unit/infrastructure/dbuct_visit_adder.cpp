// dbuct_visit_adder writes a visit increment through to the bank and mirrors it
// into the top frame's visit lump.  Mocks the frame source and the bank.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_visit_adder.hpp"
#include "value_objects/dbuct_frame.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

namespace
{

struct MockGetTopFrame
{
    MOCK_METHOD((monte_carlo::dbuct_frame<int>&), top, (), ());
};

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockSetVisits
{
    MOCK_METHOD(void, set_visits, (const int&, size_t), ());
};

}

class DbuctVisitAdderTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_frame<int>           frame_{7, 10};
    NiceMock<MockGetTopFrame>               get_top_frame;
    NiceMock<MockGetVisits>                 get_visits;
    StrictMock<MockSetVisits>               set_visits;
    monte_carlo::dbuct_visit_adder<int,
                                   MockGetTopFrame,
                                   MockGetVisits,
                                   MockSetVisits> sut{get_top_frame, get_visits, set_visits};
};

TEST_F(DbuctVisitAdderTest, AddVisitsIncrementsBankAndFrameLump)
{
    frame_.visit_lump = 2;
    ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(5));
    EXPECT_CALL(set_visits, set_visits(7, 8));

    sut.add_visits(3);

    EXPECT_EQ(frame_.visit_lump, 5u);
}
