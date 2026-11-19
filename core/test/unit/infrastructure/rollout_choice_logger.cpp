// rollout_choice_logger chooses through the rollout policy it wraps, logs that
// choice, and returns it.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/rollout_choice_logger.hpp"

using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Ref;
using ::testing::Return;
using ::testing::StrictMock;

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

struct MockRolloutChoose
{
    MOCK_METHOD(int, rollout_choose, (const stub_choice_count&, const stub_choice_at&), ());
};

struct MockPushChoice
{
    MOCK_METHOD(void, push_choice, (const int&), ());
};

using logger_t = monte_carlo::rollout_choice_logger<int, stub_choice_count, stub_choice_at,
                                                    MockRolloutChoose, MockPushChoice>;

}

class RolloutChoiceLoggerTest : public ::testing::Test
{
protected:
    NiceMock<MockRolloutChoose> rollout;
    NiceMock<MockPushChoice>    push_choice;
    logger_t                    sut{rollout, push_choice};
    stub_choice_count           count;
    stub_choice_at              at;
};

TEST_F(RolloutChoiceLoggerTest, ReturnsTheWrappedRolloutsChoice)
{
    ON_CALL(rollout, rollout_choose(_, _)).WillByDefault(Return(20));

    EXPECT_EQ(sut.rollout_choose(count, at), 20);
}

TEST_F(RolloutChoiceLoggerTest, LogsTheChoiceItReturns)
{
    ON_CALL(rollout, rollout_choose(_, _)).WillByDefault(Return(20));
    EXPECT_CALL(push_choice, push_choice(20));

    sut.rollout_choose(count, at);
}

TEST_F(RolloutChoiceLoggerTest, PassesTheChoicesThroughToTheWrappedRollout)
{
    EXPECT_CALL(rollout, rollout_choose(Ref(count), Ref(at))).WillOnce(Return(0));

    sut.rollout_choose(count, at);
}

TEST_F(RolloutChoiceLoggerTest, EachChoiceIsLoggedOnceInOrder)
{
    StrictMock<MockPushChoice> strict_push;
    logger_t                   strict_sut{rollout, strict_push};
    EXPECT_CALL(rollout, rollout_choose(_, _)).WillOnce(Return(10)).WillOnce(Return(0));
    InSequence order;
    EXPECT_CALL(strict_push, push_choice(10));
    EXPECT_CALL(strict_push, push_choice(0));

    strict_sut.rollout_choose(count, at);
    strict_sut.rollout_choose(count, at);
}
