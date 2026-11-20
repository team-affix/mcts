// choice_logger chooses through the chooser it wraps, logs that choice, and
// returns it.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/choice_logger.hpp"

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
    size_t size() const { return 2; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i); }
};

struct MockChoose
{
    MOCK_METHOD(int, choose, (const stub_choice_count&, const stub_choice_at&), ());
};

struct MockPushChoice
{
    MOCK_METHOD(void, push_choice, (const int&), ());
};

using logger_t = monte_carlo::choice_logger<int, stub_choice_count, stub_choice_at, MockChoose, MockPushChoice>;

}

class ChoiceLoggerTest : public ::testing::Test
{
protected:
    NiceMock<MockChoose>     chooser;
    NiceMock<MockPushChoice> push_choice;
    logger_t                 sut{chooser, push_choice};
    stub_choice_count        count;
    stub_choice_at           at;
};

TEST_F(ChoiceLoggerTest, ReturnsTheWrappedChoosersChoice)
{
    ON_CALL(chooser, choose(_, _)).WillByDefault(Return(1));
    EXPECT_EQ(sut.choose(count, at), 1);
}

TEST_F(ChoiceLoggerTest, LogsTheChoiceItReturns)
{
    ON_CALL(chooser, choose(_, _)).WillByDefault(Return(1));
    EXPECT_CALL(push_choice, push_choice(1));
    sut.choose(count, at);
}

TEST_F(ChoiceLoggerTest, PassesTheChoicesThroughToTheWrappedChooser)
{
    EXPECT_CALL(chooser, choose(Ref(count), Ref(at))).WillOnce(Return(0));
    sut.choose(count, at);
}

TEST_F(ChoiceLoggerTest, EachChoiceIsLoggedOnceInOrder)
{
    StrictMock<MockPushChoice> strict_push;
    logger_t                   strict_sut{chooser, strict_push};
    EXPECT_CALL(chooser, choose(_, _)).WillOnce(Return(1)).WillOnce(Return(0));
    InSequence order;
    EXPECT_CALL(strict_push, push_choice(1));
    EXPECT_CALL(strict_push, push_choice(0));
    strict_sut.choose(count, at);
    strict_sut.choose(count, at);
}
