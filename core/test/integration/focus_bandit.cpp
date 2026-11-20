// focus_value_manifest on small games with deterministic rewards: tied choices
// stop splitting the visits evenly (the first tied child takes the most),
// a real difference still wins once the band has shrunk past it, a choice
// whose subtree improves still overtakes the focus, and every child keeps
// being explored.

#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "infrastructure/focus_value_manifest.hpp"
#include "infrastructure/uct_value_manifest.hpp"
#include "infrastructure/value_bounds.hpp"
#include "infrastructure/value_table.hpp"
#include "infrastructure/visits_table.hpp"
#include "walkers.hpp"

class FocusBanditIntegrationTest : public ::testing::Test
{
protected:
    using node_t     = std::vector<int>;
    using visits_t   = monte_carlo::visits_table<node_t, path_unordered_map>;
    using value_t    = monte_carlo::value_table<node_t, double, path_unordered_map>;
    using bounds_t   = monte_carlo::value_bounds<double>;
    using focus_t    = monte_carlo::focus_value_manifest<
                          node_t, jump_t, double,
                          visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                          path_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;
    using plain_t    = monte_carlo::uct_value_manifest<
                          node_t, jump_t, double,
                          visits_t, visits_t, value_t, value_t, bounds_t, bounds_t,
                          path_walker,
                          std::vector<jump_t>, std::vector<jump_t>,
                          std::mt19937>;

    static constexpr double kC     = 4.0;
    static constexpr double kWidth = 0.1;
    static constexpr double kRate  = 0.1;
    // A bonus of 1 would keep a non-focus child's visits at about (c / (gap + 1))^2 ln N, too few for a choice whose
    // mean rises only once its own subtree concentrates (the last game here); a quarter leaves room for that.
    static constexpr double kBonus = 0.25;

    const node_t root{-1};

    // Ten arms, jumps 1 to 10.
    const std::vector<jump_t> arms{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // One choice among the arms, rewarded with the chosen arm's reward.
    template<typename IManifest>
    void play_arms(IManifest& m, const std::vector<double>& rewards, int episodes)
    {
        for (int episode = 0; episode < episodes; ++episode)
        {
            const jump_t arm = m.chooser.choose(arms, arms);
            m.delta.set_value(rewards[arm - 1]);
            m.terminator.terminate();
        }
    }

    size_t arm_visits(const visits_t& visits, jump_t arm) const
    {
        return visits.get_visits(path_walker().walk(root, arm));
    }

    // Nine tied arms worth 1 and a last arm worth 0.
    const std::vector<double> tied{1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

    visits_t     visits;
    value_t      value;
    bounds_t     bounds;
    std::mt19937 rng{5};
};

TEST_F(FocusBanditIntegrationTest, TheFirstTiedArmTakesMoreThanTwiceAnyOtherTiedArmsVisits)
{
    focus_t m(visits, visits, value, value, bounds, bounds, rng, kC, kWidth, kRate, kBonus, root);
    play_arms(m, tied, 3000);

    for (jump_t arm = 2; arm <= 9; ++arm)
        EXPECT_GT(arm_visits(visits, 1), 2 * arm_visits(visits, arm));
}

TEST_F(FocusBanditIntegrationTest, PlainUctSplitsTheTiedArmsEvenly)
{
    plain_t m(visits, visits, value, value, bounds, bounds, rng, kC, root);
    play_arms(m, tied, 3000);

    for (jump_t arm = 2; arm <= 9; ++arm)
        EXPECT_LE(arm_visits(visits, 1), arm_visits(visits, arm) + 1);
}

TEST_F(FocusBanditIntegrationTest, EveryArmKeepsBeingExplored)
{
    focus_t m(visits, visits, value, value, bounds, bounds, rng, kC, kWidth, kRate, kBonus, root);
    play_arms(m, tied, 3000);
    std::vector<size_t> before;
    for (jump_t arm : arms)
        before.push_back(arm_visits(visits, arm));

    play_arms(m, tied, 30000);
    for (jump_t arm : arms)
        EXPECT_GT(arm_visits(visits, arm), before[arm - 1]);
}

TEST_F(FocusBanditIntegrationTest, ASlightlyBetterLaterArmWinsOnceTheBandIsNarrower)
{
    // Arm 2 is better than arm 1 by 0.02 on the normalized scale; the band starts at 0.1 and shrinks as N^(-1/2),
    // so arm 1 is the focus only until about 25 visits.
    focus_t m(visits, visits, value, value, bounds, bounds, rng, kC, kWidth, 0.5, kBonus, root);
    const std::vector<double> rewards{0.98, 1.0, 0, 0, 0, 0, 0, 0, 0, 0};
    play_arms(m, rewards, 3000);

    EXPECT_GT(arm_visits(visits, 2), 2 * arm_visits(visits, 1));
}

TEST_F(FocusBanditIntegrationTest, AChoiceWhoseSubtreeImprovesOvertakesTheFocus)
{
    // The first move's arm 1 is worth 0.6 whatever follows; arm 2 is worth 1 if the second move is its arm 7 and
    // 0 otherwise, so arm 2 looks worse until its subtree finds arm 7.
    focus_t m(visits, visits, value, value, bounds, bounds, rng, kC, kWidth, kRate, kBonus, root);
    const std::vector<jump_t> first_moves{1, 2};
    for (int episode = 0; episode < 20000; ++episode)
    {
        const jump_t first  = m.chooser.choose(first_moves, first_moves);
        const jump_t second = m.chooser.choose(arms, arms);
        const bool   found  = first == 2 && second == 7;
        m.delta.set_value(first == 1 ? 0.6 : (found ? 1.0 : 0.0));
        m.terminator.terminate();
    }

    EXPECT_GT(arm_visits(visits, 2), arm_visits(visits, 1));
}

TEST_F(FocusBanditIntegrationTest, AFreshManifestPerEpisodeSearchesTheSameAsOne)
{
    visits_t kept_visits;
    value_t  kept_value;
    bounds_t kept_bounds;
    std::mt19937 kept_rng{5};
    focus_t kept(kept_visits, kept_visits, kept_value, kept_value, kept_bounds, kept_bounds, kept_rng,
                 kC, kWidth, kRate, kBonus, root);
    play_arms(kept, tied, 500);

    for (int episode = 0; episode < 500; ++episode)
    {
        focus_t fresh(visits, visits, value, value, bounds, bounds, rng, kC, kWidth, kRate, kBonus, root);
        play_arms(fresh, tied, 1);
    }

    for (jump_t arm : arms)
        EXPECT_EQ(arm_visits(visits, arm), arm_visits(kept_visits, arm));
}
