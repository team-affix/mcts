// visit_proportional_grant computes a node's grant as 1 + k * visits(node),
// truncated to an integer, so the grant is never zero.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/visit_proportional_grant.hpp"

using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

}

class VisitProportionalGrantTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits> get_visits;
};

TEST_F(VisitProportionalGrantTest, ZeroKAlwaysGrantsOne)
{
    monte_carlo::visit_proportional_grant<int, double, MockGetVisits> sut{get_visits, 0.0};
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(1000));

    EXPECT_EQ(sut.get_grant(7), 1u);
}

TEST_F(VisitProportionalGrantTest, ProductBelowOneStillGrantsOne)
{
    monte_carlo::visit_proportional_grant<int, double, MockGetVisits> sut{get_visits, 0.25};
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(3));

    EXPECT_EQ(sut.get_grant(7), 1u);
}

TEST_F(VisitProportionalGrantTest, GrantGrowsProportionallyWithVisits)
{
    monte_carlo::visit_proportional_grant<int, double, MockGetVisits> sut{get_visits, 0.5};

    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(8));
    ON_CALL(get_visits, get_visits(9)).WillByDefault(Return(20));

    EXPECT_EQ(sut.get_grant(7), 5u);
    EXPECT_EQ(sut.get_grant(9), 11u);
}

TEST_F(VisitProportionalGrantTest, FractionalProductIsFloored)
{
    monte_carlo::visit_proportional_grant<int, double, MockGetVisits> sut{get_visits, 0.5};
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(5));

    EXPECT_EQ(sut.get_grant(7), 3u);
}
