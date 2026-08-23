// uct_value_creditor layers value accumulation over visit credit for the node on
// top of the backprop path.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_value_creditor.hpp"

using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace
{

struct MockCreditVisit
{
    MOCK_METHOD(void, credit, (), ());
};

struct MockGetTopNode
{
    MOCK_METHOD(int, top, (), (const));
};

struct MockUpdateNode
{
    MOCK_METHOD(void, update, (const int&), ());
};

}

class UctValueCreditorTest : public ::testing::Test
{
protected:
    StrictMock<MockCreditVisit>     visit_creditor;
    NiceMock<MockGetTopNode>        get_top_node;
    StrictMock<MockUpdateNode>      update_node;
    monte_carlo::uct_value_creditor<MockCreditVisit,
                                    MockGetTopNode,
                                    MockUpdateNode> sut{
        visit_creditor, get_top_node, update_node};
};

TEST_F(UctValueCreditorTest, CreditVisitsThenUpdatesValueForTopNode)
{
    ON_CALL(get_top_node, top()).WillByDefault(Return(7));

    InSequence seq;
    EXPECT_CALL(visit_creditor, credit());
    EXPECT_CALL(update_node, update(7));

    sut.credit();
}
