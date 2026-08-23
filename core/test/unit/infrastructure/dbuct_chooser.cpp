// dbuct_chooser delegates to the rollout policy while in rollout, and otherwise
// dispatches a sub-budget, foresteps a new frame, and enters rollout when the
// chosen child has never been visited.

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/dbuct_chooser.hpp"
#include "value_objects/dbuct_frame.hpp"
#include "walkers.hpp"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

namespace
{

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockGetDispatches
{
    MOCK_METHOD(size_t, get_dispatches, (const int&), (const));
};

struct MockSetDispatches
{
    MOCK_METHOD(void, set_dispatches, (const int&, size_t), ());
};

struct MockComputeBatchSize
{
    MOCK_METHOD(size_t, compute_batch_size, (size_t), (const));
};

struct MockForestep
{
    MOCK_METHOD(void, forestep, (const monte_carlo::dbuct_frame<int>&), ());
};

struct MockGetTopFrame
{
    MOCK_METHOD((monte_carlo::dbuct_frame<int>&), top, (), ());
};

struct MockPolicyChoose
{
    MOCK_METHOD(int, policy_choose, (const int&, const std::vector<jump_t>&, const std::vector<jump_t>&), ());
};

struct MockRolloutChoose
{
    MOCK_METHOD(jump_t, rollout_choose, (const std::vector<jump_t>&, const std::vector<jump_t>&), ());
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

class DbuctChooserTest : public ::testing::Test
{
protected:
    monte_carlo::dbuct_frame<int> frame_{-1, 100};
    NiceMock<MockGetVisits>       get_visits;
    NiceMock<MockGetDispatches>   get_dispatches;
    StrictMock<MockSetDispatches> set_dispatches;
    NiceMock<MockComputeBatchSize> compute_batch_size;
    StrictMock<MockForestep>      forestep;
    NiceMock<MockGetTopFrame>     get_top_frame;
    position_walker               walker;
    StrictMock<MockPolicyChoose>  policy;
    StrictMock<MockRolloutChoose> rollout;
    NiceMock<MockIsInRollout>     is_in_rollout;
    StrictMock<MockEnterRollout>  enter_rollout;
    std::vector<jump_t>           jumps_{1};

    monte_carlo::dbuct_chooser<int, jump_t,
                                 MockGetVisits,
                                 MockGetDispatches,
                                 MockSetDispatches,
                                 MockComputeBatchSize,
                                 MockForestep,
                                 MockGetTopFrame,
                                 position_walker,
                                 std::vector<jump_t>,
                                 std::vector<jump_t>,
                                 MockPolicyChoose,
                                 MockRolloutChoose,
                                 MockIsInRollout,
                                 MockEnterRollout> sut{
        get_visits,
        get_dispatches,
        set_dispatches,
        compute_batch_size,
        forestep,
        get_top_frame,
        walker,
        policy,
        rollout,
        is_in_rollout,
        enter_rollout};

    void expect_tree_frame()
    {
        ON_CALL(get_top_frame, top()).WillByDefault(ReturnRef(frame_));
    }
};

TEST_F(DbuctChooserTest, RolloutPhaseDelegatesToRolloutChoose)
{
    ON_CALL(is_in_rollout, is_in_rollout()).WillByDefault(Return(true));
    EXPECT_CALL(rollout, rollout_choose(jumps_, jumps_)).WillOnce(Return(jump_t{1}));
    EXPECT_CALL(policy, policy_choose(_, _, _)).Times(0);
    EXPECT_CALL(forestep, forestep(_)).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(DbuctChooserTest, TreePhaseDispatchesAndForesteps)
{
    expect_tree_frame();
    ON_CALL(is_in_rollout, is_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_dispatches, get_dispatches(-1)).WillByDefault(Return(0));
    ON_CALL(compute_batch_size, compute_batch_size(0)).WillByDefault(Return(5));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(5));

    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(set_dispatches, set_dispatches(-1, 1));
    EXPECT_CALL(forestep, forestep(_));
    EXPECT_CALL(rollout, rollout_choose(_, _)).Times(0);
    EXPECT_CALL(enter_rollout, enter_rollout()).Times(0);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}

TEST_F(DbuctChooserTest, TreePhaseEntersRolloutOnUnvisitedChild)
{
    expect_tree_frame();
    ON_CALL(is_in_rollout, is_in_rollout()).WillByDefault(Return(false));
    ON_CALL(get_dispatches, get_dispatches(-1)).WillByDefault(Return(0));
    ON_CALL(compute_batch_size, compute_batch_size(0)).WillByDefault(Return(5));
    ON_CALL(get_visits, get_visits(0)).WillByDefault(Return(0));

    EXPECT_CALL(policy, policy_choose(-1, jumps_, jumps_)).WillOnce(Return(1));
    EXPECT_CALL(set_dispatches, set_dispatches(-1, 1));
    EXPECT_CALL(forestep, forestep(_));
    EXPECT_CALL(enter_rollout, enter_rollout()).Times(1);

    EXPECT_EQ(sut.choose(jumps_, jumps_), 1);
}
