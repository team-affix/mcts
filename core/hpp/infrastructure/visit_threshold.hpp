#ifndef VISIT_THRESHOLD_HPP
#define VISIT_THRESHOLD_HPP

#include <cstddef>

namespace monte_carlo
{

// visit_threshold
//
// A commit condition: the frontier has gathered at least a set number of
// visits since it became the frontier.

struct visit_threshold
{
    visit_threshold(size_t visits);

    bool reached(size_t visits_since_frontier) const;

private:
    size_t visits_;
};

}

#endif
