#include <iostream>
#include "test_utils.hpp"

#include "mcts.hpp"

#define FLT_CMP(x, y, threshold) (std::abs(x - y) < threshold)
#define FLT_CMP_DEFAULT_THRESHOLD 0.001
#define IS_CLOSE_TO(x, y) FLT_CMP(x, y, FLT_CMP_DEFAULT_THRESHOLD)

//////////
// helpers
//////////

double coin_collecting_game(
    monte_carlo::tree_node&    a_root,
    const std::vector<double>& a_track,
    const std::vector<int>&    a_jump_lengths,
    std::mt19937&              a_rnd_dev)
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

    // construct a reasonable exploration constant
    double l_exploration_constant = 0;
    for (double l_coin : a_track)
        if (l_coin > 0) l_exploration_constant += l_coin;

    // l_exploration_constant *= 2;

    return monte_carlo::tree_search(a_root, l_act_fxn, l_value_fxn, l_actions.size(), l_exploration_constant, a_rnd_dev);
    
}

//////////

void test_coin_collecting_game_0()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    std::mt19937                           l_mt19937(27);
    std::uniform_real_distribution<double> l_urd(-10, 10);
    
    std::vector<double>    l_track(10);
    std::generate(l_track.begin(), l_track.end(), [&l_mt19937, &l_urd] {return l_urd(l_mt19937);});
    
    std::vector<int>       l_moves{1, 2, 3};
    monte_carlo::tree_node l_root{};

    constexpr int SIMULATIONS = 10000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = coin_collecting_game(l_root, l_track, l_moves, l_mt19937);
        // LOG(l_score << std::endl);
    }
    
    // LOG(l_root.m_visits << std::endl);

    assert(IS_CLOSE_TO(l_score, 26.91873636));
    
}

void test_coin_collecting_game_1()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    std::mt19937                           l_mt19937(28);
    std::uniform_real_distribution<double> l_urd(-10, 10);
    
    std::vector<double>    l_track(10);
    std::generate(l_track.begin(), l_track.end(), [&l_mt19937, &l_urd] {return l_urd(l_mt19937);});
    
    std::vector<int>       l_moves{1, 2, 3};
    monte_carlo::tree_node l_root{};

    constexpr int SIMULATIONS = 100000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = coin_collecting_game(l_root, l_track, l_moves, l_mt19937);
        // LOG(l_score << std::endl);
    }
    
    // LOG(l_root.m_visits << std::endl);

    assert(IS_CLOSE_TO(l_score, 10.145513480));
    
}

void unit_test_main()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    TEST(test_coin_collecting_game_0);
    TEST(test_coin_collecting_game_1);
}

int main()
{
    unit_test_main();
    return 0;
}
