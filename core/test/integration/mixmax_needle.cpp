// mixmax_value_manifest on a needle game with deterministic rewards: one first
// move is worth a fixed amount whatever follows, the other is worth more only
// along one path two moves further down and nothing elsewhere. UCT reads the
// second move by its average over that mostly empty subtree; mixmax also reads
// the best path found through it, so it moves its visits to the needle sooner.

#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/mixmax_value_manifest.hpp"
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_bounds.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class MixmaxNeedleIntegrationTest : public ::testing::Test
{
protected:
    using node_t   = std::vector<int>;
    using visits_t = monte_carlo::visits_table<node_t, path_unordered_map>;
    using value_t  = monte_carlo::value_table<node_t, double, path_unordered_map>;
    using bounds_t = monte_carlo::value_bounds<double>;
    using mixmax_t = monte_carlo::mixmax_value_manifest<
                        node_t, jump_t, double,
                        visits_t, visits_t, value_t, value_t, value_t, value_t, bounds_t, bounds_t,
                        path_walker,
                        std::vector<jump_t>, std::vector<jump_t>,
                        std::mt19937>;
    using plain_t  = monte_carlo::uct_value_manifest<
                        node_t, jump_t, double,
                        visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                        path_walker,
                        std::vector<jump_t>, std::vector<jump_t>,
                        std::mt19937>;

    // Exploration wide enough that UCT also reaches the needle, so the two differ in what they do once it is found.
    static constexpr double kC        = 4.0;
    static constexpr int    kEpisodes = 20000;

    const node_t root{-1};
    const std::vector<jump_t> first_moves{1, 2};
    const std::vector<jump_t> arms{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // First move 1 is worth 0.6; first move 2 is worth 1 if the next two moves are 7 then 3, and 0 otherwise.
    template<typename IManifest>
    void play_needle(IManifest& m, int episodes)
    {
        for (int episode = 0; episode < episodes; ++episode)
        {
            const jump_t first  = m.chooser.choose(first_moves, first_moves);
            const jump_t second = m.chooser.choose(arms, arms);
            const jump_t third  = m.chooser.choose(arms, arms);
            const bool   needle = first == 2 && second == 7 && third == 3;
            m.delta.set_value(first == 1 ? 0.6 : (needle ? 1.0 : 0.0));
            m.terminator.terminate();
        }
    }

    size_t first_move_visits(jump_t move) const
    {
        return visits.get_visits(path_walker().walk(root, move));
    }

    size_t needle_visits() const
    {
        const path_walker walker;
        return visits.get_visits(walker.walk(walker.walk(walker.walk(root, 2), 7), 3));
    }

    visits_t     visits;
    value_t      value;
    value_t      best;
    bounds_t     bounds;
    std::mt19937 rng{5};
};

TEST_F(MixmaxNeedleIntegrationTest, HalfWeightOnTheBestMovesTheVisitsToTheNeedle)
{
    mixmax_t m(visits, visits, value, value, best, best, bounds, bounds, rng, kC, 0.5, root);
    play_needle(m, kEpisodes);

    EXPECT_GT(first_move_visits(2), 10 * first_move_visits(1));
    EXPECT_GT(needle_visits(), kEpisodes / 2);
}

TEST_F(MixmaxNeedleIntegrationTest, PlainUctFindsTheNeedleButStaysWithTheFixedMove)
{
    plain_t m(visits, visits, value, value, bounds, bounds, rng, kC, root);
    play_needle(m, kEpisodes);

    EXPECT_GT(needle_visits(), 0u);
    EXPECT_GT(first_move_visits(1), 10 * first_move_visits(2));
}

TEST_F(MixmaxNeedleIntegrationTest, NoWeightOnTheBestSearchesTheSameAsPlainUct)
{
    visits_t     plain_visits;
    value_t      plain_value;
    bounds_t     plain_bounds;
    std::mt19937 plain_rng{5};
    plain_t plain(plain_visits, plain_visits, plain_value, plain_value, plain_bounds, plain_bounds, plain_rng, kC,
                  root);
    play_needle(plain, 3000);

    mixmax_t m(visits, visits, value, value, best, best, bounds, bounds, rng, kC, 0.0, root);
    play_needle(m, 3000);

    for (jump_t move : first_moves)
        EXPECT_EQ(first_move_visits(move), plain_visits.get_visits(path_walker().walk(root, move)));
}

TEST_F(MixmaxNeedleIntegrationTest, TheFixedMoveKeepsBeingExplored)
{
    mixmax_t m(visits, visits, value, value, best, best, bounds, bounds, rng, kC, 1.0, root);
    play_needle(m, kEpisodes);
    const size_t before = first_move_visits(1);

    play_needle(m, kEpisodes);
    EXPECT_GT(first_move_visits(1), before);
}

TEST_F(MixmaxNeedleIntegrationTest, TheBestRewardThroughEachFirstMoveIsKept)
{
    mixmax_t m(visits, visits, value, value, best, best, bounds, bounds, rng, kC, 0.5, root);
    play_needle(m, kEpisodes);

    EXPECT_DOUBLE_EQ(best.get_value(path_walker().walk(root, 1)), 0.6);
    EXPECT_DOUBLE_EQ(best.get_value(path_walker().walk(root, 2)), 1.0);
}
