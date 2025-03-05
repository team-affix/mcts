#include <iostream>
#include "test_utils.hpp"

#include "mcts.hpp"

#define FLT_CMP(x, y, threshold) (std::abs(x - y) < threshold)
#define FLT_CMP_DEFAULT_THRESHOLD 0.001
#define IS_CLOSE_TO(x, y) FLT_CMP(x, y, FLT_CMP_DEFAULT_THRESHOLD)

//////////
// helpers
//////////

struct jump
{
    int m_amount = 0;
    bool operator<(const jump& a_other) const {return m_amount < a_other.m_amount;}
};

double simulate_coin_collecting_game(
    monte_carlo::tree_node<jump>& a_root,
    const std::vector<double>&    a_track,
    const std::vector<int>&       a_jump_lengths,
    std::mt19937&                 a_rnd_dev)
{
    // construct a reasonable exploration constant
    double l_exploration_constant = 0;
    for (double l_coin : a_track)
        if (l_coin > 0) l_exploration_constant += l_coin;

    // construct the simulation context object
    monte_carlo::simulation<jump, std::mt19937> l_sim(a_root, l_exploration_constant, a_rnd_dev);
    
    // define state
    int    l_position    = -1;
    double l_total_score = 0;

    // define actions
    std::vector<jump> l_actions;
    std::transform(a_jump_lengths.begin(), a_jump_lengths.end(), std::back_inserter(l_actions),
        [](int a_jl){return jump{a_jl};});

    // loop until end of game condition.
    while (true)
    {
        // get next action
        jump l_chosen_action = l_sim.choose(l_actions);
        // jump to new position
        l_position += l_chosen_action.m_amount;
        // terminal state check
        if (l_position >= a_track.size()) break;
        // collect coin at this position of the track
        l_total_score += a_track[l_position];
    }

    // terminate simulation with the final score
    l_sim.terminate(l_total_score);

    return l_total_score;
    
}

//////////

void test_coin_collecting_game_0()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    std::mt19937                           l_mt19937(27);
    std::uniform_real_distribution<double> l_urd(-10, 10);
    
    std::vector<double>    l_track(10);
    std::generate(l_track.begin(), l_track.end(), [&l_mt19937, &l_urd] {return l_urd(l_mt19937);});
    
    std::vector<int>             l_moves{1, 2, 3};
    monte_carlo::tree_node<jump> l_root{};

    constexpr int SIMULATIONS = 10000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = simulate_coin_collecting_game(l_root, l_track, l_moves, l_mt19937);
        LOG(l_score << std::endl);
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
    
    std::vector<int>             l_moves{1, 2, 3};
    monte_carlo::tree_node<jump> l_root{};

    constexpr int SIMULATIONS = 100000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = simulate_coin_collecting_game(l_root, l_track, l_moves, l_mt19937);
        // LOG(l_score << std::endl);
    }
    
    // LOG(l_root.m_visits << std::endl);

    assert(IS_CLOSE_TO(l_score, 10.145513480));
    
}

void test_coin_collecting_game_2()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    std::mt19937                           l_mt19937(29);
    std::uniform_real_distribution<double> l_urd(-10, 10);
    
    std::vector<double>    l_track(30);
    std::generate(l_track.begin(), l_track.end(), [&l_mt19937, &l_urd] {return l_urd(l_mt19937);});
    
    std::vector<int>             l_moves{1, 2, 3};
    monte_carlo::tree_node<jump> l_root{};

    constexpr int SIMULATIONS = 100000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = simulate_coin_collecting_game(l_root, l_track, l_moves, l_mt19937);
        // LOG(l_score << std::endl);
    }
    
    // LOG(l_root.m_visits << std::endl);

    assert(IS_CLOSE_TO(l_score, 52.135578));
    
}

void test_coin_collecting_game_3()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    std::mt19937                           l_mt19937(30);
    std::uniform_real_distribution<double> l_urd(-10, 10);
    
    std::vector<double>    l_track(30);
    std::generate(l_track.begin(), l_track.end(), [&l_mt19937, &l_urd] {return l_urd(l_mt19937);});
    
    std::vector<int>             l_moves{2, 3, 5, 7}; // prime moves only :)
    monte_carlo::tree_node<jump> l_root{};

    constexpr int SIMULATIONS = 100000;

    double l_score;

    for (int i = 0; i < SIMULATIONS; ++i)
    {
        l_score = simulate_coin_collecting_game(l_root, l_track, l_moves, l_mt19937);
        // LOG(l_score << std::endl);
    }
    
    // LOG(l_root.m_visits << std::endl);

    assert(IS_CLOSE_TO(l_score, 45.596276158));
    
}

void unit_test_main()
{
    constexpr bool ENABLE_DEBUG_LOGS = true;
    
    TEST(test_coin_collecting_game_0);
    TEST(test_coin_collecting_game_1);
    TEST(test_coin_collecting_game_2);
    TEST(test_coin_collecting_game_3);
}

int main()
{
    unit_test_main();
    return 0;
}
