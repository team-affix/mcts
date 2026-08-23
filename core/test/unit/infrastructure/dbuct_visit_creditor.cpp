// dbuct_visit_creditor turns a credit event into exactly one added visit.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_visit_creditor.hpp"

using ::testing::StrictMock;

namespace
{

struct MockAddVisits
{
    MOCK_METHOD(void, add_visits, (size_t), ());
};

}

class DbuctVisitCreditorTest : public ::testing::Test
{
protected:
    StrictMock<MockAddVisits>                    visit_adder;
    monte_carlo::dbuct_visit_creditor<MockAddVisits> sut{visit_adder};
};

TEST_F(DbuctVisitCreditorTest, CreditDelegatesSingleVisit)
{
    EXPECT_CALL(visit_adder, add_visits(1)).Times(1);
    sut.credit();
}
