#ifndef MIXMAX_VALUE_VIEW_HPP
#define MIXMAX_VALUE_VIEW_HPP

namespace monte_carlo
{

// mixmax_value_view
//
// A node's value as selection reads it under mixmax: the reward sum blended
// with the best reward, (1 - w) * sum + w * visits * best, so that dividing by
// the visits, as ucb1 does, gives (1 - w) * mean + w * best. With w = 0 it is
// the plain sum (UCT); with w = 1 a node is worth the best episode through it.
// The sum, the best and the visits are read from where they are kept.

template<
    typename INodeHandle,
    typename IFloat,
    typename IGetVisits,
    typename IGetValue,
    typename IGetBest
>
struct mixmax_value_view
{
    mixmax_value_view(IGetVisits& get_visits,
                      IGetValue&  get_value,
                      IGetBest&   get_best,
                      IFloat      best_weight);

    IFloat get_value(const INodeHandle& node) const;

private:
    IGetVisits& get_visits_;
    IGetValue&  get_value_;
    IGetBest&   get_best_;
    IFloat      best_weight_;
};

template<typename INodeHandle, typename IFloat, typename IGVis, typename IGVal, typename IGB>
mixmax_value_view<INodeHandle, IFloat, IGVis, IGVal, IGB>::mixmax_value_view(
        IGVis& get_visits,
        IGVal& get_value,
        IGB&   get_best,
        IFloat best_weight)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , get_best_(get_best)
    , best_weight_(best_weight)
{}

template<typename INodeHandle, typename IFloat, typename IGVis, typename IGVal, typename IGB>
IFloat mixmax_value_view<INodeHandle, IFloat, IGVis, IGVal, IGB>::get_value(const INodeHandle& node) const
{
    const IFloat visits = static_cast<IFloat>(get_visits_.get_visits(node));
    const IFloat sum    = get_value_.get_value(node);
    const IFloat best   = get_best_.get_value(node);
    return (1 - best_weight_) * sum + best_weight_ * visits * best;
}

}

#endif
