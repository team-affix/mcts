// uct_visit_creditor credits one visit to whichever node is on top of the
// backprop path, reading the handle itself rather than being handed one.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_visit_creditor.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace
{

struct MockGetTopNode
{
    MOCK_METHOD(int, top, (), (const));
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

class UctVisitCreditorTest : public ::testing::Test
{
protected:
    NiceMock<MockGetTopNode>  get_top_node;
    NiceMock<MockGetVisits>   get_visits;
    StrictMock<MockSetVisits> set_visits;
    monte_carlo::uct_visit_creditor<int,
                                    MockGetTopNode,
                                    MockGetVisits,
                                    MockSetVisits> sut{
        get_top_node, get_visits, set_visits};
};

TEST_F(UctVisitCreditorTest, CreditWritesBackOneMoreVisitForTopNode)
{
    ON_CALL(get_top_node, top()).WillByDefault(Return(7));
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(4));
    EXPECT_CALL(set_visits, set_visits(7, 5));

    sut.credit();
}
