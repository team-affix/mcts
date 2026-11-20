#include "infrastructure/visit_threshold.hpp"

namespace monte_carlo
{

visit_threshold::visit_threshold(size_t visits)
    : visits_(visits)
{}

bool visit_threshold::reached(size_t visits_since_frontier) const
{
    return visits_since_frontier >= visits_;
}

}
