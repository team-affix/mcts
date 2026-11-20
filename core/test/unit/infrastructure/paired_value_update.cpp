// paired_value_update updates the node with its first update, then its second.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/paired_value_update.hpp"

using ::testing::InSequence;
using ::testing::StrictMock;

namespace
{

struct MockUpdateNode
{
    MOCK_METHOD(void, update, (const int&), ());
};

}

class PairedValueUpdateTest : public ::testing::Test
{
protected:
    StrictMock<MockUpdateNode> update_node0;
    StrictMock<MockUpdateNode> update_node1;
    monte_carlo::paired_value_update<int, MockUpdateNode, MockUpdateNode> sut{update_node0, update_node1};
};

TEST_F(PairedValueUpdateTest, UpdatesTheNodeWithTheFirstThenTheSecond)
{
    InSequence in_order;
    EXPECT_CALL(update_node0, update(42));
    EXPECT_CALL(update_node1, update(42));

    sut.update(42);
}

TEST_F(PairedValueUpdateTest, EachUpdateGetsTheNodeItWasGiven)
{
    InSequence in_order;
    EXPECT_CALL(update_node0, update(1));
    EXPECT_CALL(update_node1, update(1));
    EXPECT_CALL(update_node0, update(2));
    EXPECT_CALL(update_node1, update(2));

    sut.update(1);
    sut.update(2);
}
