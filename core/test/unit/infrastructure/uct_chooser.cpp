// uct_chooser advances the cursor on every choice, but only records the node on
// the backprop path during the tree phase, and enters rollout the first time it
// steps onto an unvisited child.

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_chooser.hpp"
#include "walkers.hpp"

using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace
{

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockPolicyChoose
{
    MOCK_METHOD(int, policy_choose, (const int&, const std::vector<jump_t>&, const std::vector<jump_t>&), ());
};

struct MockRolloutChoose
{
    MOCK_METHOD(jump_t, rollout_choose, (const std::vector<jump_t>&, const std::vector<jump_t>&), ());
};

struct MockGetCurrentNode
{
    MOCK_METHOD(int, get_current_node, (), (const));
};

struct MockSetCurrentNode
{
    MOCK_METHOD(void, set_current_node, (int), ());
};

struct MockPushNode
{
    MOCK_METHOD(void, push, (const int&), ());
};

struct MockIsInRollout
{
    MOCK_METHOD(bool, is_in_rollout, (), (const));
};

struct MockEnterRollout
{
    MOCK_METHOD(void, enter_rollout, (), ());
};

}

class UctChooserTest : public ::testing::Test
{
protected:
    NiceMock<MockGetVisits>             get_visits;
    position_walker                     walker;
    StrictMock<MockPolicyChoose>        policy;
    StrictMock<MockRolloutChoose>       rollout;
    NiceMock<MockGetCurrentNode>        get_current_node;
    StrictMock<MockSetCurrentNode>      set_current_node;
    StrictMock<MockPushNode>            push_node;
    NiceMock<MockIsInRollout>           is_in_rollout;
    StrictMock<MockEnterRollout>        enter_rollout;
    std::vector<jump_t>                 jumps_{1};

    monte_carlo::uct_chooser<int, jump_t,
                             MockGetVisits,
                             position_walker,
                             std::vector<jump_t>,
                             std::vector<jump_t>,
                             MockPolicyChoose,
                             MockRolloutChoose,
                             MockGetCurrentNode,
                             MockSetCurrentNode,
                             MockPushNode,
                             MockIsInRollout,
                             MockEnterRollout> sut{
        get_visits,
        walker,
        policy,
        rollout,
        get_current_node,
        set_current_node,
        push_node,
        is_in_rollout,
        enter_rollout};
};

TEST_F(UctChooserTest, RolloutPhaseAdvancesCursorWithoutPushingPath)
{
    ON_CALL(is_in_rollout, is_in_rollout()).WillByDefault(Return(true));
    ON_CALL(get_current_node, get_current_node()).WillByDefault(Return(2));

    EXPECT_CALL(rollout, rollout_choose(jumps_, jumps_)).WillOnce(Return(jump_t{1}));
    EXPECT_CALL(set_current_node, set_current_node(3));
    EXPECT_CALL(push_node, push(_)).Times(0);
    EXPECT_CALL(policy, policy_choose(_, _, _)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(UctChooserTest, TreePhasePushesChildAndAdvancesCursor)
{
    ON_CALL(is_in_rollout, is_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_current_node, get_current_node()).WillByDefault(Return(-1));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(5));

    InSequence seq;
    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(push_node, push(0));
    EXPECT_CALL(set_current_node, set_current_node(0));
    EXPECT_CALL(enter_rollout, enter_rollout()).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(UctChooserTest, TreePhaseEntersRolloutOnUnvisitedChild)
{
    ON_CALL(is_in_rollout, is_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_current_node, get_current_node()).WillByDefault(Return(-1));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(0));

    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(push_node, push(0));
    EXPECT_CALL(set_current_node, set_current_node(0));
    EXPECT_CALL(enter_rollout, enter_rollout()).Times(1);
    EXPECT_CALL(rollout, rollout_choose(_, _)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}
