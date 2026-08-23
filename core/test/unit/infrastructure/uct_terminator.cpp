// uct_terminator drains the backprop path, crediting then popping each node, and
// then restores the cursor and path to the root so the manifest can be reused.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_terminator.hpp"

using ::testing::InSequence;
using ::testing::Return;
using ::testing::StrictMock;

namespace
{

struct MockGetNodeCount
{
    MOCK_METHOD(size_t, size, (), (const));
};

struct MockCreditor
{
    MOCK_METHOD(void, credit, (), ());
};

struct MockPopNode
{
    MOCK_METHOD(void, pop, (), ());
};

struct MockPushNode
{
    MOCK_METHOD(void, push, (const int&), ());
};

struct MockSetCurrentNode
{
    MOCK_METHOD(void, set_current_node, (int), ());
};

struct MockExitRollout
{
    MOCK_METHOD(void, exit_rollout, (), ());
};

}

class UctTerminatorTest : public ::testing::Test
{
protected:
    static constexpr int kRoot = -1;

    StrictMock<MockGetNodeCount>   get_node_count;
    StrictMock<MockCreditor>       creditor;
    StrictMock<MockPopNode>        pop_node;
    StrictMock<MockPushNode>       push_node;
    StrictMock<MockSetCurrentNode> set_current_node;
    StrictMock<MockExitRollout>    exit_rollout;
    monte_carlo::uct_terminator<int,
                                MockGetNodeCount,
                                MockCreditor,
                                MockPopNode,
                                MockPushNode,
                                MockSetCurrentNode,
                                MockExitRollout> sut{
        get_node_count, creditor, pop_node, push_node,
        set_current_node, exit_rollout, kRoot};
};

TEST_F(UctTerminatorTest, TerminateCreditsAndPopsEachNodeThenRestoresRoot)
{
    InSequence seq;
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(2));
    EXPECT_CALL(creditor, credit());
    EXPECT_CALL(pop_node, pop());
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(1));
    EXPECT_CALL(creditor, credit());
    EXPECT_CALL(pop_node, pop());
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(0));
    EXPECT_CALL(push_node, push(kRoot));
    EXPECT_CALL(set_current_node, set_current_node(kRoot));
    EXPECT_CALL(exit_rollout, exit_rollout());

    sut.terminate();
}

TEST_F(UctTerminatorTest, TerminateOnEmptyPathStillRestoresRoot)
{
    EXPECT_CALL(get_node_count, size()).WillOnce(Return(0));
    EXPECT_CALL(creditor, credit()).Times(0);
    EXPECT_CALL(pop_node, pop()).Times(0);
    EXPECT_CALL(push_node, push(kRoot));
    EXPECT_CALL(set_current_node, set_current_node(kRoot));
    EXPECT_CALL(exit_rollout, exit_rollout());

    sut.terminate();
}
