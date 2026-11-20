// committed_policy returns a committed node's choice without consulting the
// policy it wraps, and chooses through that policy everywhere else.

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/committed_policy.hpp"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Ref;
using ::testing::Return;

namespace
{

struct stub_choice_count
{
    size_t size() const { return 2; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i); }
};

struct MockFindCommitment
{
    MOCK_METHOD(std::optional<int>, find_commitment, (const int&), (const));
};

struct MockPolicyChoose
{
    MOCK_METHOD(int, policy_choose, (const int&, const stub_choice_count&, const stub_choice_at&), ());
};

using policy_t = monte_carlo::committed_policy<int, int, MockFindCommitment, MockPolicyChoose,
                                               stub_choice_count, stub_choice_at>;

}

class CommittedPolicyTest : public ::testing::Test
{
protected:
    NiceMock<MockFindCommitment> find_commitment;
    NiceMock<MockPolicyChoose>   policy;
    policy_t                     sut{find_commitment, policy};
    stub_choice_count            count;
    stub_choice_at               at;

    void SetUp() override
    {
        ON_CALL(find_commitment, find_commitment(_)).WillByDefault(Return(std::nullopt));
    }
};

TEST_F(CommittedPolicyTest, ACommittedNodeReturnsItsCommitment)
{
    ON_CALL(find_commitment, find_commitment(4)).WillByDefault(Return(std::optional<int>(1)));
    EXPECT_CALL(policy, policy_choose(_, _, _)).Times(0);

    EXPECT_EQ(sut.policy_choose(4, count, at), 1);
}

TEST_F(CommittedPolicyTest, AnUncommittedNodeChoosesThroughTheWrappedPolicy)
{
    EXPECT_CALL(policy, policy_choose(5, Ref(count), Ref(at))).WillOnce(Return(0));

    EXPECT_EQ(sut.policy_choose(5, count, at), 0);
}

TEST_F(CommittedPolicyTest, OnlyTheCommittedNodeIsFollowed)
{
    ON_CALL(find_commitment, find_commitment(4)).WillByDefault(Return(std::optional<int>(1)));
    ON_CALL(policy, policy_choose(5, _, _)).WillByDefault(Return(0));

    EXPECT_EQ(sut.policy_choose(4, count, at), 1);
    EXPECT_EQ(sut.policy_choose(5, count, at), 0);
}
