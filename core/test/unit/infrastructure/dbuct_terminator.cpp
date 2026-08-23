// dbuct_terminator credits the episode, then backsteps while the top frame has
// exhausted its budget, and finally leaves the rollout phase.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_terminator.hpp"
#include "value_objects/dbuct_frame.hpp"

using ::testing::NiceMock;
using ::testing::ReturnRef;
using ::testing::StrictMock;

namespace
{

struct MockBackstep
{
    MOCK_METHOD(void, backstep, (), ());
};

struct MockGetTopFrame
{
    MOCK_METHOD((monte_carlo::dbuct_frame<int>&), top, (), ());
};

struct MockCreditor
{
    MOCK_METHOD(void, credit, (), ());
};

struct MockExitRollout
{
    MOCK_METHOD(void, exit_rollout, (), ());
};

}

class DbuctTerminatorTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_frame<int> frame_{0, 0};
    NiceMock<MockGetTopFrame>     get_top_frame;
    StrictMock<MockBackstep>      backstep;
    StrictMock<MockCreditor>      creditor;
    StrictMock<MockExitRollout>   exit_rollout;
};

TEST_F(DbuctTerminatorTest, TerminateCreditsThenBackstepsWhileBudgetExhausted)
{
    frame_.budget     = 2;
    frame_.visit_lump = 3;
    ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));

    EXPECT_CALL(creditor, credit()).Times(1);
    EXPECT_CALL(backstep, backstep())
        .Times(2)
        .WillOnce([&] { frame_.visit_lump = 2; })
        .WillOnce([&] { frame_.visit_lump = 0; });
    EXPECT_CALL(exit_rollout, exit_rollout()).Times(1);

    monte_carlo::dbuct_terminator<MockBackstep,
                                  MockGetTopFrame,
                                  MockCreditor,
                                  MockExitRollout> sut{
        backstep, get_top_frame, creditor, exit_rollout};

    sut.terminate();
}

TEST_F(DbuctTerminatorTest, TerminateSkipsBackstepWhenCamping)
{
    frame_.budget     = 3;
    frame_.visit_lump = 1;
    ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));

    EXPECT_CALL(creditor, credit()).Times(1);
    EXPECT_CALL(backstep, backstep()).Times(0);
    EXPECT_CALL(exit_rollout, exit_rollout()).Times(1);

    monte_carlo::dbuct_terminator<MockBackstep,
                                  MockGetTopFrame,
                                  MockCreditor,
                                  MockExitRollout> sut{
        backstep, get_top_frame, creditor, exit_rollout};

    sut.terminate();
}
