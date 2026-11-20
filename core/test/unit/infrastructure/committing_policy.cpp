// committing_policy commits the frontier to the child its finder picks once the
// commit condition holds for the visits since it became the frontier, and
// otherwise chooses through the policy it wraps.

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/committing_policy.hpp"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct stub_choice_count
{
    size_t size() const { return 3; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i) * 10; }
};

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

struct MockGetFrontier
{
    MOCK_METHOD(int, frontier, (), (const));
};

struct MockGetFrontierBaseVisits
{
    MOCK_METHOD(size_t, frontier_base_visits, (), (const));
};

struct MockReachCondition
{
    MOCK_METHOD(bool, reached, (size_t), (const));
};

struct MockFindCommitChild
{
    MOCK_METHOD(std::optional<int>, find, (const int&, const stub_choice_count&, const stub_choice_at&), (const));
};

struct MockCommit
{
    MOCK_METHOD(void, commit, (const int&, const int&, const int&, size_t), ());
};

struct MockPolicyChoose
{
    MOCK_METHOD(int, policy_choose, (const int&, const stub_choice_count&, const stub_choice_at&), ());
};

using policy_t = monte_carlo::committing_policy<int, int, MockGetVisits, MockWalker, MockGetFrontier,
                                                MockGetFrontierBaseVisits, MockReachCondition, MockFindCommitChild, MockCommit,
                                                MockPolicyChoose, stub_choice_count, stub_choice_at>;

}

class CommittingPolicyTest : public ::testing::Test
{
protected:
    static constexpr int kFrontier = 4;

    NiceMock<MockGetVisits>             get_visits;
    NiceMock<MockWalker>                walker;
    NiceMock<MockGetFrontier>           get_frontier;
    NiceMock<MockGetFrontierBaseVisits> get_frontier_base_visits;
    NiceMock<MockReachCondition>        reach_condition;
    NiceMock<MockFindCommitChild>       find_commit_child;
    NiceMock<MockCommit>                commit;
    NiceMock<MockPolicyChoose>          policy;
    policy_t sut{get_visits, walker, get_frontier, get_frontier_base_visits, reach_condition, find_commit_child,
                 commit, policy};
    stub_choice_count count;
    stub_choice_at    at;

    void SetUp() override
    {
        ON_CALL(get_frontier, frontier()).WillByDefault(Return(kFrontier));
        ON_CALL(get_frontier_base_visits, frontier_base_visits()).WillByDefault(Return(30));
        ON_CALL(get_visits, get_visits(kFrontier)).WillByDefault(Return(130));
        ON_CALL(reach_condition, reached(_)).WillByDefault(Return(true));
        ON_CALL(policy, policy_choose(_, _, _)).WillByDefault(Return(20));
        for (int choice = 0; choice <= 20; choice += 10)
            ON_CALL(walker, walk(kFrontier, choice)).WillByDefault(Return(100 + choice));
        set_child_visits(30, 50, 20);
        ON_CALL(find_commit_child, find(kFrontier, _, _)).WillByDefault(Return(std::optional<int>(10)));
    }

    void set_child_visits(size_t first, size_t second, size_t third)
    {
        ON_CALL(get_visits, get_visits(100)).WillByDefault(Return(first));
        ON_CALL(get_visits, get_visits(110)).WillByDefault(Return(second));
        ON_CALL(get_visits, get_visits(120)).WillByDefault(Return(third));
    }
};

TEST_F(CommittingPolicyTest, ANodeOtherThanTheFrontierChoosesThroughTheWrappedPolicy)
{
    EXPECT_CALL(commit, commit(_, _, _, _)).Times(0);
    EXPECT_CALL(policy, policy_choose(9, _, _)).WillOnce(Return(10));

    EXPECT_EQ(sut.policy_choose(9, count, at), 10);
}

TEST_F(CommittingPolicyTest, TheConditionSeesOnlyTheVisitsSinceTheFrontierBecameTheFrontier)
{
    EXPECT_CALL(reach_condition, reached(100)).WillOnce(Return(false));

    sut.policy_choose(kFrontier, count, at);
}

TEST_F(CommittingPolicyTest, BeforeTheConditionHoldsTheFrontierChoosesThroughTheWrappedPolicy)
{
    ON_CALL(reach_condition, reached(_)).WillByDefault(Return(false));
    EXPECT_CALL(commit, commit(_, _, _, _)).Times(0);

    EXPECT_EQ(sut.policy_choose(kFrontier, count, at), 20);
}

TEST_F(CommittingPolicyTest, OnceTheConditionHoldsTheFoundChildIsCommittedWithItsVisits)
{
    EXPECT_CALL(commit, commit(kFrontier, 10, 110, 50u));
    EXPECT_CALL(policy, policy_choose(_, _, _)).Times(0);

    EXPECT_EQ(sut.policy_choose(kFrontier, count, at), 10);
}

TEST_F(CommittingPolicyTest, TheFinderIsAskedAboutTheFrontiersChoices)
{
    EXPECT_CALL(find_commit_child, find(kFrontier, ::testing::Ref(count), ::testing::Ref(at)))
        .WillOnce(Return(std::optional<int>(20)));
    EXPECT_CALL(commit, commit(kFrontier, 20, 120, 20u));

    EXPECT_EQ(sut.policy_choose(kFrontier, count, at), 20);
}

TEST_F(CommittingPolicyTest, BeforeTheConditionHoldsTheFinderIsNotAsked)
{
    ON_CALL(reach_condition, reached(_)).WillByDefault(Return(false));
    EXPECT_CALL(find_commit_child, find(_, _, _)).Times(0);

    sut.policy_choose(kFrontier, count, at);
}

TEST_F(CommittingPolicyTest, WhenTheFinderPicksNothingNothingIsCommitted)
{
    ON_CALL(find_commit_child, find(_, _, _)).WillByDefault(Return(std::nullopt));
    EXPECT_CALL(commit, commit(_, _, _, _)).Times(0);

    EXPECT_EQ(sut.policy_choose(kFrontier, count, at), 20);
}
