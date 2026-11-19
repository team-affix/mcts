// best_path_grafter exits the rollout and, when the episode's reward is at least
// the best grafted so far, replays the logged rollout choices from the top of
// the backprop path, pushing each node reached. The log is always cleared.

#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/best_path_grafter.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

namespace
{

struct MockExitRollout
{
    MOCK_METHOD(void, exit_rollout, (), ());
};

struct MockGetReward
{
    MOCK_METHOD(double, value, (), (const));
};

struct MockGetChoices
{
    MOCK_METHOD(const std::vector<int>&, choices, (), (const));
};

struct MockClearChoices
{
    MOCK_METHOD(void, clear_choices, (), ());
};

struct MockGetTopNode
{
    MOCK_METHOD(int, top, (), (const));
};

struct MockWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

struct MockPushNode
{
    MOCK_METHOD(void, push, (const int&), ());
};

using grafter_t = monte_carlo::best_path_grafter<int, int, double,
                                                 MockExitRollout,
                                                 MockGetReward,
                                                 MockGetChoices,
                                                 MockClearChoices,
                                                 MockGetTopNode,
                                                 MockWalker,
                                                 MockPushNode>;

}

class BestPathGrafterTest : public ::testing::Test
{
protected:
    static constexpr int kFrontier = 5;

    NiceMock<MockExitRollout>  exit_rollout;
    NiceMock<MockGetReward>    get_reward;
    NiceMock<MockGetChoices>   get_choices;
    NiceMock<MockClearChoices> clear_choices;
    NiceMock<MockGetTopNode>   get_top_node;
    NiceMock<MockWalker>       walker;
    NiceMock<MockPushNode>     push_node;
    grafter_t sut{exit_rollout, get_reward, get_choices, clear_choices, get_top_node, walker, push_node};

    std::vector<int> no_choices;
    std::vector<int> two_choices{1, 2};

    void SetUp() override
    {
        ON_CALL(get_choices, choices()).WillByDefault(ReturnRef(two_choices));
        ON_CALL(get_top_node, top()).WillByDefault(Return(kFrontier));
        ON_CALL(walker, walk(kFrontier, 1)).WillByDefault(Return(6));
        ON_CALL(walker, walk(6, 2)).WillByDefault(Return(8));
    }

    void graft_with_reward(double reward)
    {
        ON_CALL(get_reward, value()).WillByDefault(Return(reward));
        sut.graft();
    }
};

TEST_F(BestPathGrafterTest, TheFirstEpisodeIsGraftedHoweverLowItsReward)
{
    EXPECT_CALL(push_node, push(_)).Times(AnyNumber());
    EXPECT_CALL(push_node, push(8));

    graft_with_reward(-1e300);
}

TEST_F(BestPathGrafterTest, TheRolloutIsReplayedFromTheTopNodeInOrder)
{
    std::vector<int> three_choices{1, 2, 0};
    ON_CALL(get_choices, choices()).WillByDefault(ReturnRef(three_choices));
    ON_CALL(walker, walk(8, 0)).WillByDefault(Return(9));

    InSequence order;
    EXPECT_CALL(walker, walk(kFrontier, 1));
    EXPECT_CALL(push_node, push(6));
    EXPECT_CALL(walker, walk(6, 2));
    EXPECT_CALL(push_node, push(8));
    EXPECT_CALL(walker, walk(8, 0));
    EXPECT_CALL(push_node, push(9));

    graft_with_reward(-3.0);
}

TEST_F(BestPathGrafterTest, AWorseEpisodeIsNotGrafted)
{
    graft_with_reward(-3.0);

    EXPECT_CALL(walker, walk(_, _)).Times(0);
    EXPECT_CALL(push_node, push(_)).Times(0);
    graft_with_reward(-5.0);
}

TEST_F(BestPathGrafterTest, AnEpisodeEqualToTheBestIsGrafted)
{
    graft_with_reward(-3.0);

    EXPECT_CALL(push_node, push(_)).Times(2);
    graft_with_reward(-3.0);
}

TEST_F(BestPathGrafterTest, ABetterEpisodeIsGrafted)
{
    graft_with_reward(-3.0);

    EXPECT_CALL(push_node, push(_)).Times(2);
    graft_with_reward(-1.0);
}

TEST_F(BestPathGrafterTest, AWorseEpisodeDoesNotLowerTheBest)
{
    graft_with_reward(-3.0);
    graft_with_reward(-5.0);

    EXPECT_CALL(push_node, push(_)).Times(0);
    graft_with_reward(-4.0);
}

TEST_F(BestPathGrafterTest, ABetterEpisodeRaisesTheBest)
{
    graft_with_reward(-3.0);
    graft_with_reward(-1.0);

    EXPECT_CALL(push_node, push(_)).Times(0);
    graft_with_reward(-2.0);
}

TEST_F(BestPathGrafterTest, AnEpisodeWithoutARolloutPushesNothing)
{
    ON_CALL(get_choices, choices()).WillByDefault(ReturnRef(no_choices));

    EXPECT_CALL(walker, walk(_, _)).Times(0);
    EXPECT_CALL(push_node, push(_)).Times(0);
    graft_with_reward(-3.0);
}

TEST_F(BestPathGrafterTest, TheRolloutIsExitedBeforeAnyNodeIsWalked)
{
    EXPECT_CALL(walker, walk(_, _)).Times(AnyNumber());
    InSequence order;
    EXPECT_CALL(exit_rollout, exit_rollout());
    EXPECT_CALL(walker, walk(kFrontier, 1));

    graft_with_reward(-3.0);
}

TEST_F(BestPathGrafterTest, TheRolloutIsExitedWhenNotGrafted)
{
    graft_with_reward(-3.0);

    EXPECT_CALL(exit_rollout, exit_rollout());
    graft_with_reward(-5.0);
}

TEST_F(BestPathGrafterTest, TheLogIsClearedAfterTheReplay)
{
    EXPECT_CALL(push_node, push(_)).Times(AnyNumber());
    InSequence order;
    EXPECT_CALL(push_node, push(8));
    EXPECT_CALL(clear_choices, clear_choices());

    graft_with_reward(-3.0);
}

TEST_F(BestPathGrafterTest, TheLogIsClearedWhenNotGrafted)
{
    graft_with_reward(-3.0);

    EXPECT_CALL(clear_choices, clear_choices());
    graft_with_reward(-5.0);
}
