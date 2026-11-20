// Best-path commits through best_path_commit_value_manifest on the
// coin-collecting track: the recorded best path replays to the best score, and
// every commitment follows it. The commitments and the best path live outside
// the manifest, so they outlast it.

#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/best_path_commit_value_manifest.hpp"
#include "infrastructure/best_path_store.hpp"
#include "infrastructure/commitment_store.hpp"
#include "infrastructure/value_bounds.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class BestPathCommitTrackGameIntegrationTest : public ::testing::Test
{
protected:
    using node_t     = std::vector<int>;
    using visits_t   = monte_carlo::visits_table<node_t, path_unordered_map>;
    using value_t    = monte_carlo::value_table<node_t, double, path_unordered_map>;
    using bounds_t   = monte_carlo::value_bounds<double>;
    using commits_t  = monte_carlo::commitment_store<node_t, jump_t, path_unordered_map>;
    using best_t     = monte_carlo::best_path_store<node_t, jump_t, double, path_unordered_map>;
    using manifest_t = monte_carlo::best_path_commit_value_manifest<
                          node_t, jump_t, double,
                          visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                          commits_t, commits_t, commits_t, commits_t,
                          best_t, best_t, best_t, best_t,
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
    commits_t    commitments{node_t{-1}};
    best_t       best_path;
    std::mt19937 rng{5};

    void play()
    {
        manifest_t m(visits, visits, value, value, bounds, bounds,
                     commitments, commitments, commitments, commitments,
                     best_path, best_path, best_path, best_path,
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

    // The score of following the best path's steps from the root to the end of
    // the track.
    double replay_best_path() const
    {
        node_t node     = root;
        int    position = -1;
        double score    = 0.0;
        while (true)
        {
            const jump_t jump = best_path.find_step(node).value();
            node = path_walker().walk(node, jump);
            position += jump;
            if (position >= static_cast<int>(track.size()))
                return score;
            score += track[position];
        }
    }
};

TEST_F(BestPathCommitTrackGameIntegrationTest, TheBestPathReplaysToTheBestScore)
{
    play_many(300);

    EXPECT_DOUBLE_EQ(replay_best_path(), best_path.best_reward());
}

TEST_F(BestPathCommitTrackGameIntegrationTest, TheRootIsCommittedToTheBestPathsFirstChoice)
{
    play_many(kCommitVisits);
    const jump_t first = best_path.find_step(root).value();

    play();

    EXPECT_EQ(commitments.find_commitment(root), std::optional<jump_t>(first));
}

TEST_F(BestPathCommitTrackGameIntegrationTest, EveryCommitmentFollowsTheBestPath)
{
    play_many(4 * kCommitVisits + 1);
    ASSERT_EQ(commitments.commitment_count(), 4u);

    node_t node = root;
    while (commitments.find_commitment(node).has_value())
    {
        EXPECT_EQ(commitments.find_commitment(node), best_path.find_step(node));
        node = path_walker().walk(node, commitments.find_commitment(node).value());
    }
    EXPECT_EQ(node, commitments.frontier());
}

TEST_F(BestPathCommitTrackGameIntegrationTest, TheBestScoreNeverFallsAsCommitmentsAreMade)
{
    double previous = best_path.best_reward();
    for (size_t episode = 0; episode < 4 * kCommitVisits; ++episode)
    {
        play();
        EXPECT_GE(best_path.best_reward(), previous);
        previous = best_path.best_reward();
    }
    EXPECT_DOUBLE_EQ(replay_best_path(), best_path.best_reward());
}
