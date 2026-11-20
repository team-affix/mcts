// commitment_store keeps each committed node's choice and the frontier below the
// last commitment, with the frontier's visits when it became the frontier.

#include <optional>
#include <unordered_map>
#include <gtest/gtest.h>
#include "infrastructure/commitment_store.hpp"

class CommitmentStoreTest : public ::testing::Test
{
protected:
    monte_carlo::commitment_store<int, int, std::unordered_map> store{-1};
};

TEST_F(CommitmentStoreTest, TheRootIsTheFirstFrontierWithNoVisitsCounted)
{
    EXPECT_EQ(store.frontier(), -1);
    EXPECT_EQ(store.frontier_base_visits(), 0u);
    EXPECT_EQ(store.commitment_count(), 0u);
}

TEST_F(CommitmentStoreTest, AnUncommittedNodeHasNoCommitment)
{
    EXPECT_EQ(store.find_commitment(-1), std::nullopt);
}

TEST_F(CommitmentStoreTest, CommittingRecordsTheChoiceAndMovesTheFrontierToTheChild)
{
    store.commit(-1, 2, 7, 40);
    EXPECT_EQ(store.find_commitment(-1), std::optional<int>(2));
    EXPECT_EQ(store.frontier(), 7);
    EXPECT_EQ(store.frontier_base_visits(), 40u);
    EXPECT_EQ(store.commitment_count(), 1u);
}

TEST_F(CommitmentStoreTest, EachCommitmentIsKeptAsTheFrontierAdvances)
{
    store.commit(-1, 2, 7, 40);
    store.commit(7, 0, 9, 12);
    EXPECT_EQ(store.find_commitment(-1), std::optional<int>(2));
    EXPECT_EQ(store.find_commitment(7), std::optional<int>(0));
    EXPECT_EQ(store.find_commitment(9), std::nullopt);
    EXPECT_EQ(store.frontier(), 9);
    EXPECT_EQ(store.frontier_base_visits(), 12u);
    EXPECT_EQ(store.commitment_count(), 2u);
}
