#ifndef OPTIMAL_SCORES_HPP
#define OPTIMAL_SCORES_HPP

#include <vector>
#include "walkers.hpp"

// Dynamic-programming oracles the game suites converge against.

// DP for the coin-collecting game: maximise the SUM of coins along the path.
double optimal_cumulative_score(const std::vector<double>& track,
                                const std::vector<jump_t>& jumps);

// DP for the terminal-reward game: maximise track[last_in_bounds_position].
double optimal_last_position_score(const std::vector<double>& track,
                                   const std::vector<jump_t>& jumps);

#endif
