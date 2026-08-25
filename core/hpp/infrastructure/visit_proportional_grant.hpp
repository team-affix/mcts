#ifndef VISIT_PROPORTIONAL_GRANT_HPP
#define VISIT_PROPORTIONAL_GRANT_HPP

#include <cstddef>

namespace monte_carlo
{

// visit_proportional_grant<INodeHandle, IFloat, IGetVisits>
//
// Concrete IGetGrant policy.  Implements the formula:
//   get_grant(n) = 1 + k * visits(n)
//
// where visits(n) is the node's banked visit count and k is the proportionality
// constant supplied at construction.  The cast to size_t floors the product,
// and the leading 1 makes the grant unconditionally at least 1, so a child is
// never granted a budget of zero.
//
// Special case: k = 0 gives get_grant(n) = 1 for all n, which is equivalent to
// vanilla UCT (one simulation per sub-budget).

template<typename INodeHandle, typename IFloat, typename IGetVisits>
struct visit_proportional_grant
{
    visit_proportional_grant(IGetVisits& get_visits, IFloat k);

    size_t get_grant(const INodeHandle& h) const;

private:
    IGetVisits& get_visits_;
    IFloat      k_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename INH, typename IF, typename IGVis>
visit_proportional_grant<INH, IF, IGVis>::visit_proportional_grant(
        IGVis& get_visits,
        IF     k)
    : get_visits_(get_visits)
    , k_(k)
{}

template<typename INH, typename IF, typename IGVis>
size_t visit_proportional_grant<INH, IF, IGVis>::get_grant(const INH& h) const
{
    return 1 + static_cast<size_t>(k_ * get_visits_.get_visits(h));
}

} // namespace monte_carlo

#endif // VISIT_PROPORTIONAL_GRANT_HPP
