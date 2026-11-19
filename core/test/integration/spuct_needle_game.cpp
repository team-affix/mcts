// The needle game: a deterministic single-player game where averaging hides
// the best path. Every episode makes k_depth two-way choices. Choosing 0 first
// is safe and always scores k_safe_reward. Choosing 1 first scores
// k_needle_reward only if every later choice is 1 as well, and nothing
// otherwise; so the first move 1 averages far below k_safe_reward under random
// play even though it leads to the best score.
//
// SP-UCT backs up the best reward seen, so once one episode finds the needle,
// that path keeps its value and greedy play follows it. UCT backs up the
// average, so with the same budget greedy play stays on the safe move.

#include <cstddef>
#include <random>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/spuct_value_manifest.hpp"
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"

namespace
{

constexpr size_t k_depth         = 10;
constexpr double k_safe_reward   = 10.0;
constexpr double k_needle_reward = 100.0;

// Large enough against the rewards' scale that move 1 keeps getting visits
// while it averages next to nothing.
constexpr double k_exploration_constant = 50.0;

// Node handles number a complete binary tree: the root is 0 and choice c from
// node n leads to 2n + 1 + c, so every path is its own node.
struct heap_walker
{
    int walk(const int& node, int choice) const
    {
        return 2 * node + 1 + choice;
    }
};

using visits_t = monte_carlo::visits_table<int, std::unordered_map>;
using value_t  = monte_carlo::value_table<int, double, std::unordered_map>;

template<template<typename...> typename Manifest>
using manifest_t = Manifest<int, int, double,
                            visits_t, visits_t, value_t, value_t,
                            heap_walker,
                            std::vector<int>, std::vector<int>,
                            std::mt19937>;

double score(const std::vector<int>& moves)
{
    if (moves.front() == 0)
        return k_safe_reward;
    for (const int move : moves)
    {
        if (move != 1)
            return 0.0;
    }
    return k_needle_reward;
}

template<template<typename...> typename Manifest>
double play_once(visits_t& visits, value_t& value, std::mt19937& rng, double exploration_constant)
{
    manifest_t<Manifest> m(visits, visits, value, value, rng, exploration_constant, 0);
    const std::vector<int> choices{0, 1};

    std::vector<int> moves;
    for (size_t step = 0; step < k_depth; ++step)
        moves.push_back(m.chooser.choose(choices, choices));

    const double reward = score(moves);
    m.delta.set_value(reward);
    m.terminator.terminate();
    return reward;
}

template<template<typename...> typename Manifest>
double greedy_score_after_training(unsigned seed, size_t episodes)
{
    std::mt19937 rng(seed);
    visits_t     visits;
    value_t      value;

    for (size_t episode = 0; episode < episodes; ++episode)
        play_once<Manifest>(visits, value, rng, k_exploration_constant);

    return play_once<Manifest>(visits, value, rng, 0.0);
}

// The move greedy play makes at the root. The manifest is dropped before any
// credit, so the tables are left as they were.
template<template<typename...> typename Manifest>
int first_greedy_move(visits_t& visits, value_t& value, std::mt19937& rng)
{
    manifest_t<Manifest> m(visits, visits, value, value, rng, 0.0, 0);
    const std::vector<int> choices{0, 1};
    return m.chooser.choose(choices, choices);
}

}

// Seeds 4 and 5 are ones whose play finds the needle within the budget (others
// never reach it, under either manifest, and play stays safe).
TEST(SpuctNeedleGameTest, SpuctFindsAndKeepsTheNeedle)
{
    for (const unsigned seed : {4u, 5u})
    {
        EXPECT_EQ(greedy_score_after_training<monte_carlo::spuct_value_manifest>(seed, 5000), k_needle_reward)
            << "seed " << seed;
    }
}

TEST(SpuctNeedleGameTest, UctWithTheSameBudgetStaysOnTheSafeMove)
{
    for (const unsigned seed : {4u, 5u})
    {
        EXPECT_EQ(greedy_score_after_training<monte_carlo::uct_value_manifest>(seed, 5000), k_safe_reward)
            << "seed " << seed;
    }
}

TEST(SpuctNeedleGameTest, OneNeedleEpisodeIsEnoughForTheRootToPreferIt)
{
    std::mt19937 rng(4);
    visits_t     visits;
    value_t      value;

    // Train until an episode finds the needle. The root's child for move 1 then
    // holds the needle reward as its best, however poor its other episodes, and
    // greedy play's first move is 1.
    double reward = 0.0;
    for (size_t episode = 0; episode < 5000 && reward != k_needle_reward; ++episode)
        reward = play_once<monte_carlo::spuct_value_manifest>(visits, value, rng, k_exploration_constant);

    ASSERT_EQ(reward, k_needle_reward);
    const heap_walker walker;
    EXPECT_EQ(value.get_value(walker.walk(0, 1)), k_needle_reward);
    EXPECT_EQ(value.get_value(walker.walk(0, 0)), k_safe_reward);
    EXPECT_EQ(first_greedy_move<monte_carlo::spuct_value_manifest>(visits, value, rng), 1);
}
