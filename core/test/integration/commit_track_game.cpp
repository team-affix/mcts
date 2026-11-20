// Commitments through commit_value_manifest on the coin-collecting track:
// nothing is committed before the frontier gathers its visits; then the root is
// committed to its most visited child, every later episode takes that child,
// the frontier moves one level down per threshold of visits, and search goes on
// below it. Commitments live outside the manifest, so they outlast it.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/commit_value_manifest.hpp"
#include "infrastructure/commitment_store.hpp"
#include "infrastructure/value_bounds.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class CommitTrackGameIntegrationTest : public ::testing::Test
{
protected:
    using node_t     = std::vector<int>;
    using visits_t   = monte_carlo::visits_table<node_t, path_unordered_map>;
    using value_t    = monte_carlo::value_table<node_t, double, path_unordered_map>;
    using bounds_t   = monte_carlo::value_bounds<double>;
    using store_t    = monte_carlo::commitment_store<node_t, jump_t, path_unordered_map>;
    using manifest_t = monte_carlo::commit_value_manifest<
                          node_t, jump_t, double,
                          visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                          store_t, store_t, store_t, store_t,
                          path_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;

    static constexpr size_t kCommitVisits = 200;

    const std::vector<double> track{3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, 6.0, -2.0, 7.0};
    const std::vector<jump_t> jumps{1, 2, 3};
    const node_t              root{-1};

    visits_t     visits;
    value_t      value;
    bounds_t     bounds;
    store_t      commitments{node_t{-1}};
    std::mt19937 rng{5};

    // One coin-collecting episode through a manifest made for it.
    void play()
    {
        manifest_t m(visits, visits, value, value, bounds, bounds,
                     commitments, commitments, commitments, commitments,
                     rng, 1.0, kCommitVisits, root);
        int    position = -1;
        double score    = 0.0;
        while (true)
        {
            position += m.chooser.choose(jumps, jumps);
            if (position >= static_cast<int>(track.size()))
                break;
            score += track[position];
        }
        m.delta.set_value(score);
        m.terminator.terminate();
    }

    void play_many(size_t episodes)
    {
        for (size_t episode = 0; episode < episodes; ++episode)
            play();
    }

    size_t child_visits(const node_t& node, jump_t jump) const
    {
        return visits.get_visits(path_walker().walk(node, jump));
    }
};

TEST_F(CommitTrackGameIntegrationTest, NothingIsCommittedBeforeTheRootGathersItsVisits)
{
    play_many(kCommitVisits - 1);

    EXPECT_EQ(commitments.commitment_count(), 0u);
    EXPECT_EQ(commitments.frontier(), root);
}

TEST_F(CommitTrackGameIntegrationTest, TheRootIsCommittedToItsMostVisitedChild)
{
    play_many(kCommitVisits);
    size_t most_visits = 0;
    for (jump_t jump : jumps)
        most_visits = std::max(most_visits, child_visits(root, jump));

    play();

    ASSERT_TRUE(commitments.find_commitment(root).has_value());
    const jump_t committed = commitments.find_commitment(root).value();
    EXPECT_EQ(child_visits(root, committed), most_visits + 1);
    EXPECT_EQ(commitments.frontier(), path_walker().walk(root, committed));
}

TEST_F(CommitTrackGameIntegrationTest, EveryLaterEpisodeTakesTheCommittedChild)
{
    play_many(kCommitVisits + 1);
    const jump_t committed = commitments.find_commitment(root).value();
    std::vector<size_t> before;
    for (jump_t jump : jumps)
        before.push_back(child_visits(root, jump));

    play_many(50);

    for (size_t index = 0; index < jumps.size(); ++index)
    {
        const size_t expected = before[index] + (jumps[index] == committed ? 50 : 0);
        EXPECT_EQ(child_visits(root, jumps[index]), expected);
    }
}

TEST_F(CommitTrackGameIntegrationTest, TheFrontierMovesDownOneLevelPerThreshold)
{
    play_many(3 * kCommitVisits + 1);

    EXPECT_EQ(commitments.commitment_count(), 3u);
    EXPECT_EQ(commitments.frontier().size(), 4u);
}

TEST_F(CommitTrackGameIntegrationTest, SearchGoesOnBelowTheFrontier)
{
    play_many(kCommitVisits + 1);
    const node_t frontier = commitments.frontier();
    size_t below_before = 0;
    for (jump_t jump : jumps)
        below_before += child_visits(frontier, jump);

    play_many(50);

    size_t below_after = 0;
    size_t children_tried = 0;
    for (jump_t jump : jumps)
    {
        below_after += child_visits(frontier, jump);
        children_tried += child_visits(frontier, jump) > 0;
    }
    EXPECT_EQ(below_after, below_before + 50);
    EXPECT_GT(children_tried, 1u);
}
