// dbuct_value_creditor layers value accumulation over visit credit: it credits
// the visit first, then adds the delta for the top value frame's handle.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_value_creditor.hpp"
#include "value_objects/dbuct_value_frame.hpp"

using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

namespace
{

struct MockCreditVisit
{
    MOCK_METHOD(void, credit, (), ());
};

struct MockGetTopValueFrame
{
    MOCK_METHOD((monte_carlo::dbuct_value_frame<int, double>&), top, (), ());
};

struct MockAddValue
{
    MOCK_METHOD(void, add_value, (double), ());
};

struct MockGetValueDelta
{
    MOCK_METHOD(double, get_value_delta, (const int&), (const));
};

}

class DbuctValueCreditorTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_value_frame<int, double> value_frame_{9};
    StrictMock<MockCreditVisit>                 visit_creditor;
    NiceMock<MockGetTopValueFrame>              get_top_value_frame;
    StrictMock<MockAddValue>                    value_adder;
    NiceMock<MockGetValueDelta>                 value_delta;
    monte_carlo::dbuct_value_creditor<MockCreditVisit,
                                      MockGetTopValueFrame,
                                      MockAddValue,
                                      MockGetValueDelta> sut{
        visit_creditor, get_top_value_frame, value_adder, value_delta};
};

TEST_F(DbuctValueCreditorTest, CreditVisitsThenAddsDelta)
{
    ON_CALL(get_top_value_frame, top()).WillByDefault(ReturnRef(value_frame_));
    ON_CALL(value_delta, get_value_delta(9)).WillByDefault(Return(2.5));

    InSequence seq;
    EXPECT_CALL(visit_creditor, credit());
    EXPECT_CALL(value_adder, add_value(2.5));

    sut.credit();
}
