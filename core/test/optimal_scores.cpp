#include "optimal_scores.hpp"

#include <algorithm>
#include <limits>

double optimal_cumulative_score(const std::vector<double>& track,
                                const std::vector<jump_t>& jumps)
{
    int n = (int)track.size();
    std::vector<double> dp(n, 0.0);

    for (int pos = n - 1; pos >= 0; --pos)
    {
        double best = -std::numeric_limits<double>::infinity();
        for (jump_t j : jumps)
        {
            int next = pos + j;
            best = std::max(best, (next < n) ? track[next] + dp[next] : 0.0);
        }
        dp[pos] = best;
    }

    double best = -std::numeric_limits<double>::infinity();
    for (jump_t j : jumps)
    {
        int next = -1 + j;
        best = std::max(best, (next < n) ? track[next] + dp[next] : 0.0);
    }
    return best;
}

double optimal_last_position_score(const std::vector<double>& track,
                                   const std::vector<jump_t>& jumps)
{
    int n = (int)track.size();
    std::vector<double> dp(n, -std::numeric_limits<double>::infinity());

    for (int pos = n - 1; pos >= 0; --pos)
    {
        for (jump_t j : jumps)
        {
            int    next      = pos + j;
            double candidate = (next >= n) ? track[pos] : dp[next];
            dp[pos]          = std::max(dp[pos], candidate);
        }
    }

    double best = -std::numeric_limits<double>::infinity();
    for (jump_t j : jumps)
    {
        int next = -1 + j;
        if (next >= 0 && next < n)
            best = std::max(best, dp[next]);
    }
    return best;
}
