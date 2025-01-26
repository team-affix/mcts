#include <iostream>
#include "test_utils.hpp"

#include "mcts.hpp"

//////////
// helpers
//////////

double coin_collecting_game(
    monte_carlo::tree_node&    a_root,
    const std::vector<double>& a_track,
    const std::vector<int>&    a_jump_lengths)
{
    // define state
    int    l_position    = -1;
    double l_total_score = 0;

    // define actions
    const std::vector<int>& l_actions = a_jump_lengths;

    auto l_act_fxn = [&a_track, &l_position, &l_total_score, &l_actions](size_t a_chosen_action)
    {
        // jump to new position
        l_position += l_actions[a_chosen_action];
        // terminal state check
        if (l_position >= a_track.size())
            // we have reached the track's end
            return (size_t)0;
        // collect coin at this position of the track
        l_total_score += a_track[l_position];
        // return the number of available actions.
        return l_actions.size();
    };

    auto l_value_fxn = [&l_total_score]
    {
        return l_total_score;
    };

    return monte_carlo::tree_search(a_root, l_act_fxn, l_value_fxn, l_actions.size());
    
}

//////////

void test_coin_collecting_game_0()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    std::vector<double> l_track{0.5, -2, 1.67, 5.4, -10, 4.9, 1.2, -4.3};
    std::vector<int>    l_moves{1, 2, 3};
    monte_carlo::tree_node l_root{};

    constexpr int SIMULATIONS = 10000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = coin_collecting_game(l_root, l_track, l_moves);
        LOG(l_score << std::endl);
    }
    
    LOG(l_root.m_visits << std::endl);
    
}

void unit_test_main()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    TEST(test_coin_collecting_game_0);
}

int main()
{
    unit_test_main();
    return 0;
}
